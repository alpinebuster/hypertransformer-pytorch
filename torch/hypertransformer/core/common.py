"""Datatypes, classes and functions used across `ca_supp`."""

import dataclasses
import os
import time

from typing import Any, Callable, Optional, Text, Union, \
    TypedDict
from typing_extensions import Annotated

from absl import flags
from absl import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    global_step: int = 0

    summary_writer: Optional[util.MultiFileWriter] = None
    per_step_fn: Optional[Callable[[torch.Tensor, Any, Any], None]] = None

    checkpoint_dir: str = FLAGS.train_log_dir
    checkpoint_suffix: str = "model"

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
        self.global_step = data["step"]
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


def train(train_config: TrainConfig, state: TrainState, run_options=None) -> None:
    """Train loop."""
    sess = tf.get_default_session()

    step = sess.run(state.global_step)
    first_step = step
    starting_time = time.time()

    def _save():
        state.saver.save(
            sess,
            os.path.join(FLAGS.train_log_dir, state.checkpoint_suffix),
            write_meta_graph=False,
            global_step=step,
        )

    while step <= train_config.train_steps:
        if state.step_initializer is not None:
            sess.run(state.step_initializer, options=run_options)

        to_run = {
            "step": state.global_step,
            "train": state.train_op,
            "update": state.update_op,
        }
        if state.tensors_to_eval is not None:
            to_run["tensors"] = state.tensors_to_eval
        if (step - first_step) % 100 == 1:
            logging.info(
                "Step: %d, Time per step: %f",
                step,
                (time.time() - starting_time) / (step - first_step),
            )

        if (
            train_config.steps_between_saves > 0
            and step % train_config.steps_between_saves == 0
        ):
            _save()

        if step % train_config.small_summaries_every == 0 and _is_not_empty(
            state.small_summaries
        ):
            to_run["small"] = state.small_summaries

        if step % train_config.large_summaries_every == 0 and _is_not_empty(
            state.large_summaries
        ):
            to_run["large"] = state.large_summaries

        results = sess.run(to_run, options=run_options)
        if state.per_step_fn:
            state.per_step_fn(step, sess, results)

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
    """Initializes the training loop.

    Args:
      state: Training state.

    Returns:
      True if checkpoint was found and false otherwise.
    """
    if state.summary_writer is None:
        state.summary_writer = util.MultiFileWriter(state.checkpoint_dir)

    restored = state.load_latest()
    if not restored:
        print("No checkpoint found.")
    else:
        print(f"Restored from step {state.global_step}")
    return restored
