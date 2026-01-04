"""Evaluation library."""

import copy

import dataclasses
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypertransformer.core import common
from hypertransformer.core import common_ht
from hypertransformer.core import layerwise
from hypertransformer.core import train_lib

# In some evaluation scenarios we may want to run multiple "CNN batches"
# for the same meta-training Transformer batch, which means that we have
# to fix Transformer batch and update it only occasionally.
#
# This is done in `make_train_samples` that returns:
# (a) cached Transformer images (or original tensor if `same_batch` is False);
# (b) cached Transformer labels;
# (c) operation updating Transformer image/label cache (or None if `same_batch`
#     parameter is False).
TrainSamples = Tuple[torch.Tensor, torch.Tensor, Optional[tf.Operation]]

# Type returned by `evaluate_dataset`:
# (a) a dictionary mapping a task number to a list of accuracies for all
#     batches;
# (b) a list of mean accuracies for all tasks.
Accuracies = Tuple[dict[int, list[float]], list[float]]

# Either a list of tf.Variables or a function that returns such a list.
VarList = Union[list[tf.Variable], Callable[..., list[tf.Variable]]]


@dataclasses.dataclass
class EvaluationConfig:
    num_task_evals: int
    num_eval_batches: int
    load_vars: Optional[VarList] = None


def dataset_with_custom_labels(
    model_config: common_ht.LayerwiseModelConfig,
    dataset_config: common_ht.DatasetConfig,
) -> tuple[common_ht.DatasetSamples, Optional[list[int]]]:
    """Returns a dataset with a controlled label set (should be reshuffled)."""
    custom_labels = copy.copy(dataset_config.use_label_subset)
    dataset_config = dataclasses.replace(
        dataset_config, use_label_subset=lambda: custom_labels
    )
    dataset, _ = train_lib.make_dataset(
        model_config=model_config,
        data_config=dataset_config,
        shuffle_labels=False,
    )
    return dataset, custom_labels


def make_train_samples(dataset: common_ht.DatasetSamples, same_batch=True):
    """Makes input samples for training a baseline model."""
    train_images = dataset.transformer_images
    train_labels = dataset.transformer_labels
    assign_op = None

    if same_batch:
        batch_size = train_images.shape[0]
        train_images = tf.get_variable(
            "train_images",
            shape=train_images.shape,
            dtype=train_images.dtype,
            initializer=tf.zeros_initializer,
            trainable=False,
        )
        train_labels = tf.get_variable(
            "train_labels",
            shape=(batch_size,),
            dtype=train_labels.dtype,
            initializer=tf.zeros_initializer,
            trainable=False,
        )
        assign_op = tf.group(
            tf.assign(train_images, dataset.transformer_images),
            tf.assign(train_labels, dataset.transformer_labels),
        )

    return train_images, train_labels, assign_op


def evaluate_dataset(
    custom_labels,
    dataset: common_ht.DatasetSamples,
    assign_op,
    outputs: common.ModelOutputs,
    eval_config: EvaluationConfig
) -> Accuracies:
    """Runs evaluation loop for a specific dataset."""
    test_accs = {}
    all_accs = []

    for task_num in range(eval_config.num_task_evals):
        if custom_labels:
            np.random.shuffle(custom_labels)

        # Assign op should be executed last for us to have the same augmentation
        # and labels for both Transformer and CNN samples
        if assign_op is not None:
            sess.run(assign_op)

        accs = []
        for _ in range(eval_config.num_eval_batches):
            accs.append(sess.run(outputs.accuracy))
        test_accs[task_num] = accs
        all_accs.append(np.mean(accs))

    return test_accs, all_accs


def apply_layerwise_model(
    model: layerwise.LayerwiseModel,
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples
):
    """Applies a layerwise model to a dataset."""
    with tf.variable_scope("model"):
        weight_blocks = model.train(
            dataset.transformer_images,
            dataset.transformer_labels,
            mask=dataset.transformer_masks,
        )
        predictions = model.evaluate(dataset.cnn_images, weight_blocks=weight_blocks)

    pred_labels = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    accuracy = tf.cast(tf.math.equal(dataset.cnn_labels, pred_labels), tf.float32)
    accuracy = tf.reduce_sum(accuracy) / model_config.num_cnn_samples

    return common.ModelOutputs(predictions=pred_labels, accuracy=accuracy)
