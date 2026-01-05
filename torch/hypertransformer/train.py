"""Training binary."""

import os
import functools
from typing import Optional

from absl import app, flags, logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hypertransformer import common_flags
from hypertransformer import eval_model_flags  # pylint:disable=unused-import

from hypertransformer.core import common
from hypertransformer.core import common_ht
from hypertransformer.core import layerwise
from hypertransformer.core import layerwise_defs  # pylint:disable=unused-import
from hypertransformer.core import train_lib
from hypertransformer.core import util

FLAGS = flags.FLAGS


# ------------------------------------------------------------
#   Configurations
# ------------------------------------------------------------


def make_train_config():
    return common.TrainConfig(
        train_steps=FLAGS.train_steps,
        steps_between_saves=FLAGS.steps_between_saves,
    )


def make_optimizer_config():
    return common.OptimizerConfig(
        learning_rate=FLAGS.learning_rate,
        lr_decay_steps=FLAGS.learning_rate_decay_steps,
        lr_decay_rate=FLAGS.learning_rate_decay_rate,
    )


def _common_model_config():
    """Returns common ModelConfig parameters."""
    return {
        "num_transformer_samples": FLAGS.samples_transformer,
        "num_cnn_samples": FLAGS.samples_cnn,
        "num_labels": FLAGS.num_labels,
        "image_size": FLAGS.image_size,
        "cnn_model_name": FLAGS.cnn_model_name,
        "embedding_dim": FLAGS.embedding_dim,
        "cnn_dropout_rate": FLAGS.cnn_dropout_rate,
        "use_decoder": FLAGS.use_decoder,
        "add_trainable_weights": FLAGS.add_trainable_weights,
        "var_reg_weight": FLAGS.weight_variation_regularization,
        "transformer_activation": FLAGS.transformer_activation,
        "transformer_nonlinearity": FLAGS.transformer_nonlinearity,
        "cnn_activation": FLAGS.cnn_activation,
        "default_num_channels": FLAGS.default_num_channels,
        "shared_fe_dropout": FLAGS.shared_fe_dropout,
        "fe_dropout": FLAGS.fe_dropout,
    }

def make_layerwise_model_config():
    """Makes 'layerwise' model config."""
    if not FLAGS.num_layerwise_features:
        num_features = None
    else:
        num_features = int(FLAGS.num_layerwise_features)

    if FLAGS.lw_weight_allocation == "spatial":
        weight_allocation = common_ht.WeightAllocation.SPATIAL
    elif FLAGS.lw_weight_allocation == "output":
        weight_allocation = common_ht.WeightAllocation.OUTPUT_CHANNEL
    else:
        raise ValueError(
            f"Unknown `lw_weight_allocation` flag value "
            f'"{FLAGS.lw_weight_allocation}"'
        )

    return common_ht.LayerwiseModelConfig(
        feature_layers=2,
        query_key_dim_frac=FLAGS.lw_key_query_dim,
        value_dim_frac=FLAGS.lw_value_dim,
        internal_dim_frac=FLAGS.lw_inner_dim,
        num_layers=FLAGS.num_layers,
        heads=FLAGS.heads,
        kernel_size=common_flags.KERNEL_SIZE.value,
        stride=common_flags.STRIDE.value,
        dropout_rate=FLAGS.dropout_rate,
        num_features=num_features,
        nonlinear_feature=FLAGS.lw_use_nonlinear_feature,
        weight_allocation=weight_allocation,
        generate_bn=FLAGS.lw_generate_bn,
        generate_bias=FLAGS.lw_generate_bias,
        shared_feature_extractor=FLAGS.shared_feature_extractor,
        shared_features_dim=FLAGS.shared_features_dim,
        separate_bn_vars=FLAGS.separate_evaluation_bn_vars,
        shared_feature_extractor_padding=FLAGS.shared_feature_extractor_padding,
        label_smoothing=FLAGS.label_smoothing,
        generator=FLAGS.layerwise_generator,
        train_heads=FLAGS.warmup_steps > 0,
        max_prob_remove_unlabeled=FLAGS.max_prob_remove_unlabeled,
        max_prob_remove_labeled=FLAGS.max_prob_remove_labeled,
        number_of_trained_cnn_layers=(common_flags.NUMBER_OF_TRAINED_CNN_LAYERS.value),
        skip_last_nonlinearity=FLAGS.transformer_skip_last_nonlinearity,
        l2_reg_weight=FLAGS.l2_reg_weight,
        logits_feature_extractor=FLAGS.logits_feature_extractor,
        shared_head_weight=common_flags.SHARED_HEAD_WEIGHT.value,
        **_common_model_config(),
    )


def make_dataset_config(dataset_spec=""):
    if not dataset_spec:
        dataset_spec = FLAGS.train_dataset
    dataset, label_set = util.parse_dataset_spec(dataset_spec)
    if label_set is None:
        label_set = list(range(FLAGS.use_labels))
    return common_ht.DatasetConfig(
        dataset_name=dataset,
        use_label_subset=label_set,
        ds_split="train",
        data_dir=FLAGS.data_dir,
        rotation_probability=FLAGS.rotation_probability,
        smooth_probability=FLAGS.smooth_probability,
        contrast_probability=FLAGS.contrast_probability,
        resize_probability=FLAGS.resize_probability,
        negate_probability=FLAGS.negate_probability,
        roll_probability=FLAGS.roll_probability,
        angle_range=FLAGS.angle_range,
        rotate_by_90=FLAGS.random_rotate_by_90,
        per_label_augmentation=FLAGS.per_label_augmentation,
        cache_path=FLAGS.data_numpy_dir,
        balanced_batches=FLAGS.balanced_batches,
        shuffle_labels_seed=FLAGS.shuffle_labels_seed,
        apply_image_augmentations=FLAGS.apply_image_augmentations,
        augment_individually=FLAGS.augment_images_individually,
        num_unlabeled_per_class=FLAGS.unlabeled_samples_per_class,
    )


def _default(new: float, default: float):
    return new if new >= 0 else default

def make_test_dataset_config(dataset_spec=""):
    if not dataset_spec:
        dataset_spec = FLAGS.test_dataset
    dataset, label_set = util.parse_dataset_spec(dataset_spec)
    if label_set is None:
        raise ValueError("Test dataset should specify a set of labels.")
    return common_ht.DatasetConfig(
        dataset_name=dataset,
        use_label_subset=label_set,
        ds_split=FLAGS.test_split,
        data_dir=FLAGS.data_dir,
        rotation_probability=_default(
            FLAGS.test_rotation_probability, FLAGS.rotation_probability
        ),
        smooth_probability=_default(
            FLAGS.test_smooth_probability, FLAGS.smooth_probability
        ),
        contrast_probability=_default(
            FLAGS.test_contrast_probability, FLAGS.contrast_probability
        ),
        resize_probability=_default(
            FLAGS.test_resize_probability, FLAGS.resize_probability
        ),
        negate_probability=_default(
            FLAGS.test_negate_probability, FLAGS.negate_probability
        ),
        roll_probability=_default(FLAGS.test_roll_probability, FLAGS.roll_probability),
        angle_range=_default(FLAGS.test_angle_range, FLAGS.angle_range),
        rotate_by_90=FLAGS.test_random_rotate_by_90,
        per_label_augmentation=FLAGS.test_per_label_augmentation,
        balanced_batches=FLAGS.balanced_batches,
        shuffle_labels_seed=FLAGS.shuffle_labels_seed,
        cache_path=FLAGS.data_numpy_dir,
        apply_image_augmentations=False,
        num_unlabeled_per_class=FLAGS.unlabeled_samples_per_class,
    )


# ------------------------------------------------------------
#   Training State
# ------------------------------------------------------------


def _make_optimizer(optim_config: common.OptimizerConfig, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optim_config.learning_rate,
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=optim_config.lr_decay_steps,
    #     gamma=optim_config.lr_decay_rate,
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            optim_config.lr_decay_rate
            ** (step / optim_config.lr_decay_steps)
        ),
    )

    return optimizer, scheduler


def _make_train_op(optimizer, loss, train_vars=None):
    return optimizer.minimize(
        tf.reduce_mean(loss), global_step=global_step, var_list=train_vars
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
    labels: torch.Tensor,
    predictions: torch.Tensor,
    heads: list[torch.Tensor],
    global_step: int,
    label_smoothing: float = 0.
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


# ------------------------------------------------------------
#   Model
# ------------------------------------------------------------


def create_layerwise_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    test_dataset: common_ht.DatasetSamples,
    model_state: train_lib.ModelState,
    optim_config: common.OptimizerConfig,
):
    """Creates a hierarchical Transformer-CNN model."""
    logging.info("Building the model")

    model = layerwise.build_model(
        model_config.cnn_model_name,
        model_config=model_config,
    )

    weight_blocks = model._train(
        dataset.transformer_images,
        dataset.transformer_labels,
        mask=dataset.transformer_masks,
        mask_random_samples=True,
        enable_fe_dropout=True,
    )
    predictions = model._evaluate(
        dataset.cnn_images,
        weight_blocks=weight_blocks,
    )

    heads = []
    if model_config.train_heads:
        outputs = model.layer_outputs.values()
        # layer_outputs[layer.name] => (inputs, head)
        heads = [output[1] for output in outputs if output[1] is not None]

    test_weight_blocks = model._train(
        test_dataset.transformer_images,
        test_dataset.transformer_labels,
        mask=test_dataset.transformer_masks,
    )
    test_predictions = model._evaluate(
        test_dataset.cnn_images,
        weight_blocks=test_weight_blocks,
    )

    labels = tf.one_hot(dataset.cnn_labels, depth=model_config.num_labels)
    pred_labels = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    num_cnn_samples = tf.cast(tf.shape(dataset.cnn_labels)[0], tf.float32)

    def _acc(pred):
        accuracy = tf.cast(tf.math.equal(dataset.cnn_labels, pred), tf.float32)
        return tf.reduce_sum(accuracy) / num_cnn_samples

    accuracy = _acc(pred_labels)
    head_preds = [tf.cast(tf.argmax(head, axis=-1), tf.int32) for head in heads]
    head_accs = [_acc(pred) for pred in head_preds]

    test_pred_labels = tf.cast(tf.argmax(test_predictions, axis=-1), tf.int32)
    test_accuracy = tf.cast(
        tf.math.equal(test_dataset.cnn_labels, test_pred_labels), tf.float32
    )
    test_accuracy = tf.reduce_sum(test_accuracy) / num_cnn_samples

    summaries = []
    reg_losses = tf.losses.get_losses(
        loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES
    )
    if reg_losses:
        summaries.append(
            tf.summary.scalar("loss/regularization", tf.reduce_sum(reg_losses))
        )

    shared_head_loss, shared_head_acc = model.shared_head_outputs.values()

    model_state.loss, _, warmup_weights = _make_loss(labels, predictions, heads)
    summaries.append(tf.summary.scalar("loss/ce", model_state.loss))
    if reg_losses:
        model_state.loss += tf.reduce_sum(reg_losses)
    _, optimizer = _make_optimizer(optim_config, global_step)

    if shared_head_loss is not None:
        if model_config.shared_head_weight > 0.0:
            weighted_head_loss = shared_head_loss * model_config.shared_head_weight
            model_state.loss += weighted_head_loss
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

    return common.TrainState(
        model=None,
        train_op=_make_train_op(optimizer, model_state.loss),
        step_initializer=(dataset.randomize_fn, test_dataset.randomize_fn),
        large_summaries=[],
        small_summaries=summaries
        + [
            tf.summary.scalar("accuracy/accuracy", accuracy),
            tf.summary.scalar("accuracy/test_accuracy", test_accuracy),
            tf.summary.scalar("loss/loss", model_state.loss),
        ],
    )


def create_shared_feature_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    test_dataset: common_ht.DatasetSamples,
    model_state: train_lib.ModelState,
    optim_config: common.OptimizerConfig,
):
    """Creates an image feature extractor model for pre-training."""
    del test_dataset
    logging.info("Building the model...")

    model = layerwise.build_model(
        model_config.cnn_model_name,
        model_config=model_config,
        dataset=dataset,
    )

    shared_head_loss, shared_head_acc = model.shared_head_outputs.values()
    assert shared_head_loss is not None
    _, optimizer = _make_optimizer(optim_config, global_step)
    model_state.loss = shared_head_loss

    return common.TrainState(
        train_op=_make_train_op(optimizer, model_state.loss),
        step_initializer=dataset.randomize_fn,
        large_summaries=[],
        small_summaries=[
            tf.summary.scalar("loss/shared_head_loss", shared_head_loss),
            tf.summary.scalar("accuracy/shared_head_accuracy", shared_head_acc),
        ],
    )


# ------------------------------------------------------------
#   Model Loading
# ------------------------------------------------------------


def restore_shared_features(
    model: nn.Module,
    checkpoint: str = common_flags.RESTORE_SHARED_FEATURES_FROM.value,
) -> Optional[nn.Module]:
    """Restores shared feature extractor variables from a checkpoint."""
    if not checkpoint:
        return None

    # The parameter names in the current model
    model_state: dict[str, torch.Tensor] = model.state_dict()
    # Get shared features / head
    shared_keys = [
        k for k in model_state.keys()
        if "model.shared_features" in k
        or "loss.shared_head" in k
    ]    
    if not shared_keys:
        return None

    loaded_vars = util.load_variables(
        checkpoint,
        var_list=shared_keys,
        map_location="cpu",
    )
    model.load_state_dict(loaded_vars, strict=False)

    return model


# ------------------------------------------------------------
#   Main Entrance
# ------------------------------------------------------------


def train(
    train_config: common.TrainConfig,
    optimizer_config: common.OptimizerConfig,
    dataset_config: common_ht.DatasetConfig,
    test_dataset_config: common_ht.DatasetConfig,
    layerwise_model_config: common_ht.LayerwiseModelConfig,
):
    """Main function training the model."""
    model_state = train_lib.ModelState()
    logging.info("Creating the dataset...")

    numpy_arr = train_lib.make_numpy_array(
        data_config=dataset_config,
        batch_size=layerwise_model_config.num_transformer_samples,
    )
    dataset = train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=layerwise_model_config,
        data_config=dataset_config,
    )
    test_dataset = train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=layerwise_model_config,
        data_config=test_dataset_config,
    )
    args = {
        "dataset": dataset,
        "test_dataset": test_dataset,
        "model_state": model_state,
        "optim_config": optimizer_config,
    }

    if common_flags.PRETRAIN_SHARED_FEATURE.value:
        create_model = functools.partial(
            create_shared_feature_model,
            model_config=layerwise_model_config,
        )
    else:
        create_model = functools.partial(
            create_layerwise_model,
            model_config=layerwise_model_config,
        )

    logging.info("Training")
    train_state = create_model(**args)

    init_op = restore_shared_features(
        model=train_state.model,
    )
    restored = common.init_training(train_state)
    if not restored and init_op is not None:
        sess.run(init_op)
    common.train(train_config, train_state)


def main(argv):
    # The command-line parameters have been parsed by absl.
    if len(argv) > 1:
        del argv

    logging.info(f"FLAGS: {FLAGS.flag_values_dict()}")

    gpus: str = FLAGS.gpus
    if gpus is None or gpus == "all":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    util.print_gpu_detailed_info()

    train(
        train_config=make_train_config(),
        optimizer_config=make_optimizer_config(),
        dataset_config=make_dataset_config(),
        test_dataset_config=make_test_dataset_config(),
        layerwise_model_config=make_layerwise_model_config(),
    )


if __name__ == "__main__":
    app.run(main)
