"""Datatypes, classes and functions used across `ca_supp`."""

import functools
import dataclasses
import os
import time

from typing import cast, TYPE_CHECKING, Callable, Optional, \
    TypedDict, Union
from typing_extensions import Annotated

from absl import flags, logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from hypertransformer.core import util
from hypertransformer import common_flags

if TYPE_CHECKING:
    from hypertransformer.core.layerwise import LayerwiseModel
    from hypertransformer.core import train_lib
    from hypertransformer.core import common_ht

FLAGS = flags.FLAGS

# Padding in image summaries.
IMAGE_PADDING = 4
BENCHMARK_STEPS = 100
# Number of images to save into the summary.
NUM_OUTPUT_IMAGES = 4
KEEP_CHECKPOINT_EVERY_N_HOURS = 3

field = dataclasses.field


class Sample(TypedDict):
    image: Annotated[np.ndarray, "(C, H, W)"]
    label: int | np.integer

class Batch(TypedDict):
    image: Annotated[torch.Tensor, "(B, C, H, W)"]
    label: Annotated[torch.Tensor, "(B,)"]

class BatchNumpy(TypedDict):
    image: Annotated[np.ndarray, "(B, C, H, W)"]
    label: Annotated[np.ndarray, "(B,)"]


@dataclasses.dataclass
class TrainConfig:
    """Training configuration."""

    train_steps: int = 10000
    steps_between_saves: int = 1000
    small_summaries_every: int = 100
    large_summaries_every: int = 500


@dataclasses.dataclass
class TrainState:
    """Model state."""
    model: Union["LayerwiseModel", DDP]
    model_state: "train_lib.ModelState"
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    global_step: int = 0

    checkpoint_dir: str = "logs"
    checkpoint_suffix: str = "model"

    summary_writer: Optional[util.MultiFileWriter] = None
    small_summaries: dict[str, torch.Tensor] = field(default_factory=dict)
    large_summaries: dict[str, torch.Tensor] = field(default_factory=dict)

    def save(self):
        if FLAGS.ddp and dist.get_rank() != 0:
            return

        logging.info(f"Saving checkpoint at step {self.global_step}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.checkpoint_dir,
            f"{self.checkpoint_suffix}-{self.global_step}.pt"
        )
        unwrapped_model = util.unwrap_model(self.model)
        unwrapped_model = cast("LayerwiseModel", unwrapped_model)
        torch.save({
            # Machines without a GPU can still use `torch.load` to load it
            # No reliance on CUDA version
            "model": {
                k: v.detach().cpu()
                for k, v in unwrapped_model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load_latest(self) -> tuple[bool, str]:
        ckpt_path = util.latest_checkpoint(self.checkpoint_dir)
        if not ckpt_path:
            return False, str(ckpt_path)

        data = torch.load(ckpt_path, map_location="cpu")

        assert not isinstance(self.model, DDP)
        self.model.load_state_dict(data["model"], strict=False)
        self.optimizer.load_state_dict(data["optimizer"])
        self.global_step = int(data.get("global_step", 0))

        return True, str(ckpt_path)


@dataclasses.dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    learning_rate: float = 1e-3
    lr_decay_steps: int = 10000
    lr_decay_rate: float = 1.0


@dataclasses.dataclass
class ModelOutputs:
    """Model outputs."""

    predictions: torch.Tensor
    accuracy: torch.Tensor
    labels: Optional[torch.Tensor] = None
    test_accuracy: Optional[torch.Tensor] = None


def _make_warmup_loss(
    loss_heads: list[torch.Tensor],
    loss_prediction: torch.Tensor,
    global_step: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Uses head losses to build aggregate loss cycling through them.

    Assume there are:
       1) Multiple intermediate heads (loss_heads)
       2) One final head (loss_prediction)
       3) A global training step (global_step)

    The goal is:
       1) Early training phase: Mainly rely on shallow/early heads.
       2) Mid-training phase: Transition supervision linearly between adjacent heads.
       3) Late training phase: Only the final head remains active.

    At any moment:
       1) At most, only two heads are "active."
       2) Weights are defined by a triangular function that changes linearly.

    ---

    e.g.:
       1) Given:
          - warmup_steps = 9000
          - num_heads = 3

       2) Then:
          - steps_per_stage = warmup_steps / num_heads = 3000

       3) This means:
          - Head 0: centered at step = 0
          - Head 1: centered at step = 3000
          - Head 2: centered at step = 6000

       4) Finally, the last head starts to take over completely from step = 9000.
    """
    num_heads = len(loss_heads)
    # The warmup time length corresponding to each head
    steps_per_stage = FLAGS.warmup_steps / num_heads
    loss = torch.tensor(0., device=loss_prediction.device)
    weights = []

    # The following code ends up returning just the true model head loss
    # after `global_step` reaches `warmup_steps`.

    for stage, head_loss in enumerate(loss_heads):
        target_steps = stage * steps_per_stage
        norm_step_dist = torch.abs(global_step - target_steps) / steps_per_stage
        # This weight starts at 0 and peaks reaching 1 at `target_steps`. It then
        # decays linearly to 0 and stays 0.
        # weight = max(0, 1 - norm_step_dist)
        weight = torch.clamp(1. - norm_step_dist, min=0.)
        weights.append(weight)
        loss += weight * head_loss

    target_steps = num_heads * steps_per_stage
    norm_step_dist = 1. + (global_step - target_steps) / steps_per_stage
    norm_step_dist = torch.relu(norm_step_dist)
    # Weight for the actual objective linearly grows after the final layer head
    # peaks and then stays equal to 1.
    # weight = min(1, norm_step_dist)
    weight = torch.clamp(norm_step_dist, max=1.0)
    weights.append(weight)
    loss += weight * loss_prediction

    return loss, weights

def _make_loss(
    labels: torch.Tensor, # [B,]
    predictions: torch.Tensor, # [B, num_classes]
    heads: list[torch.Tensor], # list[(B, num_classes)]
    global_step: int,
    label_smoothing: float = 0.,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Makes a full loss including head 'warmup' losses."""
    losses: list[torch.Tensor] = []

    for head in heads + [predictions]:
        head_loss = F.cross_entropy(
            head,
            labels,
            label_smoothing=label_smoothing,
        )
        losses.append(head_loss)
    if len(losses) == 1:
        return losses[0], losses, [torch.tensor(1.0, device=losses[0].device)]

    loss, warmup_weights = _make_warmup_loss(losses[:-1], losses[-1], global_step)
    return loss, losses, warmup_weights

def train(
    train_config: TrainConfig,
    model_config: "common_ht.LayerwiseModelConfig",
    state: TrainState,
    batch_provider: tuple[
        Callable[[], "common_ht.DatasetSamples"],
        Callable[[], "common_ht.DatasetSamples"]
    ],
) -> None:
    """Train loop.

    Args:
       train_config: Contains train_steps, steps_between_saves, small/large summary freqs
       state: TrainState (holds model/optimizer/step/etc.)
       batch_provider: Callable that returns a dict containing the inputs needed.
          Example return value for layerwise model:
            {
              "support_images": Tensor(Bs, C, H, W),
              "support_labels": Tensor(Bs,),
              "query_images": Tensor(Bq, C, H, W),
              "query_labels": Tensor(Bq,),
                ...
            }
    """
    assert state.summary_writer is not None, "Must run `common.init_training(state)` before start training!"

    local_rank = None
    global_rank = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if FLAGS.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    state.model = state.model.to(device)
    if not FLAGS.ddp or global_rank == 0:
        for name, param in state.model.named_parameters():
            logging.info(f"{name}, device: {param.device}, grad: {param.grad is None}")
    state.model.dataset.to(device)

    # DDP
    if FLAGS.ddp:
        # torch.autograd.set_detect_anomaly(True)
        state.model = DDP(
            state.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False, # Speed up
        )

    start_step = state.global_step
    starting_time = time.time()

    while state.global_step <= train_config.train_steps:
        add_scalar = functools.partial(
            state.summary_writer.add_scalar,
            global_step=state.global_step,
        )
        add_scalar_ddp = functools.partial(
            util.add_scalar_ddp,
            add_scalar_fn=add_scalar,
        )
        add_scalars_ddp = functools.partial(
            util.add_scalars_ddp,
            add_scalar_fn=add_scalar,
        )

        train_batch = batch_provider[0]()
        test_batch = batch_provider[1]()

        train_batch.to(device)
        test_batch.to(device)

        unwrapped_model = util.unwrap_model(state.model)
        unwrapped_model = cast("LayerwiseModel", unwrapped_model)
        state.model.train()
        total_loss = torch.tensor(0., device=device)
        if common_flags.PRETRAIN_SHARED_FEATURE.value:
            _ = state.model(
                train_batch.transformer_images,
                train_batch.transformer_labels,
                mask=train_batch.transformer_masks,
                mask_random_samples=True,
                enable_fe_dropout=True,
                only_shared_feature=True,
            )
            shared_head_loss, shared_head_acc = unwrapped_model.shared_head_outputs.values()
            assert shared_head_loss is not None and shared_head_acc is not None
            total_loss = shared_head_loss

            state.small_summaries["loss/shared_head_loss"] = shared_head_loss
            state.small_summaries["accuracy/shared_head_accuracy"] = shared_head_acc
        else:
            predictions = state.model(
                train_batch.transformer_images,
                train_batch.transformer_labels,
                train_batch.cnn_images,
                mask=train_batch.transformer_masks,
                mask_random_samples=True,
                enable_fe_dropout=True,
                deterministic_inference=False,
            )

            heads = []
            if model_config.train_heads:
                outputs = unwrapped_model.layer_outputs.values()
                # layer_outputs[layer.name] => (inputs, head)
                heads = [output[1] for output in outputs if output[1] is not None]

            labels = train_batch.cnn_labels.long()
            # Train accuracy
            pred_labels = torch.argmax(predictions, dim=-1) # (B,)

            def _acc(pred):
                accuracy = (pred == train_batch.cnn_labels).float().sum()
                return accuracy / train_batch.cnn_labels.numel()

            accuracy = _acc(pred_labels)
            head_preds = [torch.argmax(head, dim=-1) for head in heads]
            head_accs = [_acc(pred) for pred in head_preds]
            state.small_summaries["accuracy/accuracy"] = accuracy

            total_loss, _, warmup_weights = _make_loss(
                labels,
                predictions,
                heads,
                state.global_step,
            )
            add_scalar_ddp("loss/ce", total_loss)

            reg_losses = unwrapped_model.regularization_loss()
            if reg_losses:
                add_scalar_ddp("loss/regularization", reg_losses)
                total_loss += reg_losses

            shared_head_loss, shared_head_acc = unwrapped_model.shared_head_outputs.values()
            if shared_head_loss is not None and model_config.shared_head_weight > 0.:
                weighted_head_loss = shared_head_loss * model_config.shared_head_weight
                total_loss += weighted_head_loss

                add_scalar_ddp("loss/shared_head_loss", shared_head_loss)
                add_scalar_ddp("loss/weighted_shared_head_loss", weighted_head_loss)

            for head_id, acc in enumerate(head_accs):
                add_scalar_ddp(f"accuracy/head-{head_id+1}", acc)
            for head_id, warmup_weight in enumerate(warmup_weights[:-1]):
                add_scalar_ddp(f"warmup_weights/head-{head_id+1}", warmup_weight)
            if heads:
                add_scalar_ddp("warmup_weights/main", warmup_weights[-1])

            if shared_head_acc is not None and model_config.shared_head_weight > 0.:
                add_scalar_ddp("accuracy/shared_head_accuracy", shared_head_acc[-1])

            state.small_summaries["loss/loss"] = total_loss

        # Backward + Step
        state.optimizer.zero_grad()
        total_loss.backward()
        state.model_state.loss = total_loss.detach().item()
        state.optimizer.step()
        assert state.scheduler is not None
        state.scheduler.step()

        # Test accuracy
        if not common_flags.PRETRAIN_SHARED_FEATURE.value:
            state.model.eval()
            with torch.no_grad():
                test_predictions = state.model(
                    test_batch.transformer_images,
                    test_batch.transformer_labels,
                    test_batch.cnn_images,
                    mask=test_batch.transformer_masks,
                    deterministic_inference=False,
                )
            test_pred_labels = torch.argmax(test_predictions, dim=-1)
            test_accuracy = (
                (test_pred_labels == test_batch.cnn_labels)
                .float()
                .sum()
                / train_batch.cnn_labels.numel()
            )
            state.small_summaries["accuracy/test_accuracy"] = test_accuracy

        if (state.global_step - start_step) % 100 == 1 and (not FLAGS.ddp or global_rank == 0):
            logging.info(
                f"{util.get_ddp_msg()}"
                "Step: %d, Time per step: %f",
                state.global_step,
                (time.time() - starting_time) / (state.global_step - start_step),
            )

        if (
            train_config.steps_between_saves > 0
            and state.global_step % train_config.steps_between_saves == 0
        ):
            state.save()

        if state.global_step % train_config.small_summaries_every == 0:
            add_scalars_ddp(state.small_summaries)
        if state.global_step % train_config.large_summaries_every == 0:
            add_scalars_ddp(state.large_summaries)

        state.global_step += 1

    # Final save
    state.save()


def init_training(state: TrainState) -> bool:
    """Initializes the training loop by creating writer if needed
    and try restoring latest checkpoint.

    Returns:
       True if checkpoint was found and False otherwise.
    """
    if state.summary_writer is None:
        state.summary_writer = util.MultiFileWriter(state.checkpoint_dir)

    restored, ckpt_path = state.load_latest()
    if not restored:
        logging.info(f"{util.get_ddp_msg()}No checkpoint found.")
    else:
        logging.info(f"{util.get_ddp_msg()}Restored from {ckpt_path}, step {state.global_step}")
    return restored
