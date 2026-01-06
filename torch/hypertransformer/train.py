"""Training binary."""

import os
import functools
from typing import Optional

from absl import app, flags, logging

import torch
import torch.nn as nn
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
        shared_input_dim=FLAGS.shared_input_dim,
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
#   Model
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


def create_layerwise_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    model_state: train_lib.ModelState,
    optim_config: common.OptimizerConfig,
):
    """Creates a hierarchical Transformer-CNN model."""
    logging.info("Building the model...")

    model = layerwise.build_model(
        model_config.cnn_model_name,
        model_config=model_config,
        dataset=dataset,
    )
    optimizer, scheduler = _make_optimizer(optim_config, model)

    return common.TrainState(
        model=model,
        model_state=model_state,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def create_shared_feature_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    model_state: train_lib.ModelState,
    optim_config: common.OptimizerConfig,
):
    """Creates an image feature extractor model for pre-training."""
    logging.info("Building the model...")

    model = layerwise.build_model(
        model_config.cnn_model_name,
        model_config=model_config,
        dataset=dataset,
    )
    optimizer, scheduler = _make_optimizer(optim_config, model)

    return common.TrainState(
        model=model,
        model_state=model_state,
        optimizer=optimizer,
        scheduler=scheduler,
    )


# ------------------------------------------------------------
#   Model Loading
# ------------------------------------------------------------


def restore_shared_features(
    model: "layerwise.LayerwiseModel",
    checkpoint: Optional[str] = None,
) -> Optional["layerwise.LayerwiseModel"]:
    """Restores shared feature extractor variables from a checkpoint."""
    checkpoint = checkpoint or common_flags.RESTORE_SHARED_FEATURES_FROM.value
    if not checkpoint:
        return None

    # The parameter names in the current model
    model_state: dict[str, torch.Tensor] = model.state_dict()
    # Get shared features / head
    shared_keys = [
        k for k in model_state.keys()
        if k.startswith("shared_feature_extractor.") or k.startswith("shared_head.")
    ]    
    if not shared_keys:
        return None

    loaded_vars = util.load_variables(
        checkpoint,
        var_list=shared_keys,
        map_location="cpu",
    )
    # `strict=False` → Load only what exists
    model.load_state_dict(loaded_vars, strict=False)

    return model


# ------------------------------------------------------------
#   Meta Train Entrance
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

    # ALL data
    numpy_arr = train_lib.make_numpy_array(
        data_config=dataset_config,
        batch_size=layerwise_model_config.num_transformer_samples,
    )
    # Generating different episodes at initialization & per step
    train_batch_provider = lambda: train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=layerwise_model_config,
        data_config=dataset_config,
    )
    test_batch_provider = lambda: train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=layerwise_model_config,
        data_config=test_dataset_config,
    )

    args = {
        "dataset": train_batch_provider(),
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

    logging.info("Training...")
    state = create_model(**args)

    restored = common.init_training(state)
    if not restored:
        model = restore_shared_features(
            model=state.model,
        )
        if model:
            state.model = model

    common.train(
        train_config,
        layerwise_model_config,
        state,
        batch_provider=(train_batch_provider, test_batch_provider),
    )


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
