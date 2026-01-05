"""Datatypes, classes and functions used across `ca_supp`."""

import dataclasses
import os
import time

from typing import TYPE_CHECKING, Any, Callable, Optional, \
    TypedDict
from typing_extensions import Annotated

from absl import flags, logging
import numpy as np
import torch
import torch.nn.functional as F

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
    steps_between_saves: int = 5000
    small_summaries_every: int = 100
    large_summaries_every: int = 500


@dataclasses.dataclass
class TrainState:
    """Model state."""
    model: "LayerwiseModel"
    model_state: "train_lib.ModelState"
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    global_step: int = 0
    per_step_fn: Optional[Callable[[int, "TrainState", dict[str, Any]], None]] = None
    tensors_to_eval: Optional[Any] = None

    checkpoint_dir: str = FLAGS.train_log_dir
    checkpoint_suffix: str = "model"

    summary_writer: Optional[util.MultiFileWriter] = None
    # Optional: functions that produce summaries given (state, results)
    # Each summary function returns a dict[name -> value] for scalars and/or images
    small_summaries: list[Callable[["TrainState", dict[str, Any]], dict[str, Any]]] = field(default_factory=list)
    large_summaries: list[Callable[["TrainState", dict[str, Any]], dict[str, Any]]] = field(default_factory=list)

    def save(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.checkpoint_dir,
            f"{self.checkpoint_suffix}_{self.global_step}.pt"
        )
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
        }, path)

    def load_latest(self) -> bool:
        restore_path = os.path.dirname(
            os.path.join(self.checkpoint_dir, self.checkpoint_suffix)
        )
        ckpts = util.latest_checkpoint(restore_path)
        if not ckpts:
            return False
        ckpt = max(ckpts, key=os.path.getmtime)
        data = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.global_step = int(data.get("step", 0))
        return True


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


def _is_not_empty(tensor):
    if isinstance(tensor, torch.Tensor):
        return True
    return (
        tensor is not None and len(tensor) > 0
    )

def _make_warmup_loss(
    loss_heads: list[torch.Tensor],
    loss_prediction: torch.Tensor,
    global_step: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Uses head losses to build aggregate loss cycling through them."""
    # The warmup period is broken into a set of "head activation periods".
    # Each period, one head weight is linearly growing, while the previous
    # head weight goes down.
    # Basically, each moment of time only two heads are active and the active
    # heads slide towards the final layer.
    num_heads = len(loss_heads)
    steps_per_stage = FLAGS.warmup_steps / num_heads
    loss = 0
    weights = []

    # The following code ends up returning just the true model head loss
    # after `global step` reaches `warmup_steps`.

    for stage, head_loss in enumerate(loss_heads):
        target_steps = stage * steps_per_stage
        norm_step_dist = torch.abs(global_step - target_steps) / steps_per_stage
        # This weight starts at 0 and peaks reaching 1 at `target_steps`. It then
        # decays linearly to 0 and stays 0.
        # weight = max(0, 1 - norm_step_dist)
        weight = torch.clamp(1.0 - norm_step_dist, min=0.0)
        weights.append(weight)
        loss += weight * head_loss

    target_steps = num_heads * steps_per_stage
    norm_step_dist = 1.0 + (global_step - target_steps) / steps_per_stage
    norm_step_dist = torch.relu(norm_step_dist)
    # Weight for the actual objective linearly grows after the final layer head
    # peaks and then stays equal to 1.
    # weight = min(1, norm_step_dist)
    weight = torch.clamp(norm_step_dist, max=1.0)
    weights.append(weight)
    loss += weight * loss_prediction

    return loss, weights

def _make_loss(
    labels: torch.Tensor, # (B,)
    predictions: torch.Tensor, # (B, num_classes)
    heads: list[torch.Tensor], # list[(B, num_classes)]
    global_step: int,
    label_smoothing: float = 0.,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Makes a full loss including head 'warmup' losses.

    Returns:
       loss: Scalar tensor
       losses: List of scalar tensors
       warmup_weights: List of scalar tensors
    """
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

    loss, wamup_weights = _make_warmup_loss(losses[:-1], losses[-1], global_step)
    return loss, losses, wamup_weights

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
    device = next(state.model.parameters()).device
    start_step = state.global_step
    starting_time = time.time()

    def _save():
        logging.info(f"Saving checkpoint at step {state.global_step}")
        state.save()

    while state.global_step <= train_config.train_steps:
        train_batch = batch_provider[0]()
        test_batch = batch_provider[1]()

        if common_flags.PRETRAIN_SHARED_FEATURE.value:
            _ = state.model(
                train_batch.transformer_images,
                train_batch.transformer_labels,
                mask=train_batch.transformer_masks,
                mask_random_samples=True,
                enable_fe_dropout=True,
                only_shared_feature=True,
            )
            shared_head_loss, shared_head_acc = state.model.shared_head_outputs.values()
            assert shared_head_loss is not None
            state.model_state.loss = shared_head_loss
            # Add to tensorboard
            assert state.summary_writer is not None, "Must run `common.init_training(state)` before start training!"
            state.summary_writer.add_scalar(
                "loss/shared_head_loss",
                shared_head_loss,
                state.global_step,
            )
            state.summary_writer.add_scalar(
                "accuracy/shared_head_accuracy",
                shared_head_acc,
                state.global_step,
            )
        else:
            predictions = state.model(
                train_batch.transformer_images,
                train_batch.transformer_labels,
                train_batch.cnn_images,
                mask=train_batch.transformer_masks,
                mask_random_samples=True,
                enable_fe_dropout=True,
            )

            heads = []
            if model_config.train_heads:
                outputs = state.model.layer_outputs.values()
                # layer_outputs[layer.name] => (inputs, head)
                heads = [output[1] for output in outputs if output[1] is not None]

            test_predictions = state.model(
                test_batch.transformer_images,
                test_batch.transformer_labels,
                test_batch.cnn_images,
                mask=test_batch.transformer_masks,
            )

            labels = train_batch.cnn_labels.long()
            # Train accuracy
            pred_labels = torch.argmax(predictions, dim=-1) # (B,)
            num_cnn_samples: int = train_batch.cnn_labels.numel()

            def _acc(pred):
                accuracy = (pred == train_batch.cnn_labels).float().sum()
                return accuracy / num_cnn_samples

            accuracy = _acc(pred_labels)
            head_preds = [torch.argmax(head, dim=-1) for head in heads]
            head_accs = [_acc(pred) for pred in head_preds]

            # Test accuracy
            test_pred_labels = torch.argmax(test_predictions, dim=-1)
            test_accuracy = (
                (test_pred_labels == test_batch.cnn_labels)
                .float()
                .sum()
                / num_cnn_samples
            )

            state.model_state.loss, _, warmup_weights = _make_loss(
                labels,
                predictions,
                heads,
                state.global_step,
            )

            # Add to tensorboard
            summaries = []
            summaries.append(tf.summary.scalar("loss/ce", state.model_state.loss))

            reg_losses = tf.losses.get_losses(
                loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES
            )
            if reg_losses:
                summaries.append(
                    tf.summary.scalar("loss/regularization", tf.reduce_sum(reg_losses))
                )
                state.model_state.loss += tf.reduce_sum(reg_losses)

            shared_head_loss, shared_head_acc = state.model.shared_head_outputs.values()
            if shared_head_loss is not None:
                if model_config.shared_head_weight > 0.0:
                    weighted_head_loss = shared_head_loss * model_config.shared_head_weight
                    state.model_state.loss += weighted_head_loss
                    summaries.append(
                        tf.summary.scalar("loss/shared_head_loss", shared_head_loss)
                    )
                    summaries.append(
                        tf.summary.scalar("loss/weighted_shared_head_loss", weighted_head_loss)
                    )

            for head_id, acc in enumerate(head_accs):
                summaries.append(tf.summary.scalar(f"accuracy/head-{head_id+1}", acc))
            for head_id, warmup_weight in enumerate(warmup_weights[:-1]):
                summaries.append(
                    tf.summary.scalar(f"warmup_weights/head-{head_id+1}", warmup_weight)
                )
            if heads:
                summaries.append(tf.summary.scalar("warmup_weights/main", warmup_weights[-1]))

            if shared_head_acc is not None and model_config.shared_head_weight > 0.0:
                summaries.append(
                    tf.summary.scalar("accuracy/shared_head_accuracy", shared_head_acc)
                )


        to_run = {
            "step": state.global_step,
            "train": state.train_op,
            "update": state.update_op,
        }
        if state.tensors_to_eval is not None:
            to_run["tensors"] = state.tensors_to_eval
        if (state.global_step - start_step) % 100 == 1:
            logging.info(
                "Step: %d, Time per step: %f",
                state.global_step,
                (time.time() - starting_time) / (state.global_step - start_step),
            )

        if (
            train_config.steps_between_saves > 0
            and state.global_step % train_config.steps_between_saves == 0
        ):
            _save()

        if state.global_step % train_config.small_summaries_every == 0 and _is_not_empty(
            state.small_summaries
        ):
            to_run["small"] = state.small_summaries

        if state.global_step % train_config.large_summaries_every == 0 and _is_not_empty(
            state.large_summaries
        ):
            to_run["large"] = state.large_summaries

        results = sess.run(to_run, options=run_options)
        if state.per_step_fn:
            state.per_step_fn(state.global_step, sess, results)

        step = results["step"]

        if "small" in results:
            state.summary_writer.add_summary(
                util.normalize_tags(results["small"]), global_step=step
            )
        if "large" in results:
            state.summary_writer.add_summary(
                util.normalize_tags(results["large"]), global_step=step
            )

        state.global_step += 1

    # Final save.
    _save()


def init_training(state: TrainState) -> bool:
    """Initializes the training loop by creating writer if needed
    and try restoring latest checkpoint.

    Returns:
       True if checkpoint was found and False otherwise.
    """
    if state.summary_writer is None:
        state.summary_writer = util.MultiFileWriter(state.checkpoint_dir)

    restored = state.load_latest()
    if not restored:
        logging.info("No checkpoint found.")
    else:
        logging.info(f"Restored from step {state.global_step}")
    return restored
