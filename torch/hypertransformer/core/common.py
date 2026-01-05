"""Datatypes, classes and functions used across `ca_supp`."""

import dataclasses
import os
import time

from typing import Any, Callable, Optional, Union, \
    TypedDict
from typing_extensions import Annotated

from absl import flags, logging
import numpy as np
import torch

from hypertransformer.core import util

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
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    step_initializer: Union[
        Callable[[], None],
        tuple[Callable[[], None], ...],
    ]
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
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


def train(
    train_config: TrainConfig,
    state: TrainState,
    batch_provider: Callable[[], dict[str, Any]],
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
        if state.step_initializer is not None:
            if callable(state.step_initializer):
                state.step_initializer()
            else:
                for fn in state.step_initializer:
                    fn()

        batch = batch_provider()


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
