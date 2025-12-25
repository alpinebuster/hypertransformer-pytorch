"""Core implementations of "layerwise" models."""

import dataclasses
import functools
import math

from typing import Callable, Dict, List, Optional

import tensorflow.compat.v1 as tf # pyright: ignore[reportMissingImports] # pylint:disable=import-error

from hypertransformer.core import common_ht
from hypertransformer.core.common_ht import LayerwiseModelConfig
from hypertransformer.core import feature_extractors
from hypertransformer.core import transformer
from hypertransformer.core import util

GAMMA_SCALE = 1.0
BETA_SCALE = 1.0
GAMMA_BIAS = 0.0
BETA_BIAS = 0.0

models = {}

HeadBuilder = Callable[..., "BaseCNNLayer"]


@dataclasses.dataclass
class GeneratedWeights:
    weight_blocks: List[List[tf.Tensor]]
    head_weight_blocks: Dict[str, List[tf.Tensor]]
    shared_features: Optional[tf.Tensor] = None


def build_model(name: str, model_config: common_ht.LayerwiseModelConfig) -> "LayerwiseModel":
    model_fn = models[name]
    return model_fn(model_config=model_config)


def get_remove_probability(max_probability: float):
    """Returns a random probability between 0 and `max_probability`."""
    return tf.random.uniform(shape=(), dtype=tf.float32) * max_probability


def get_l2_regularizer(weight=None):
    if weight is None or weight == 0.0:
        return None
    return tf.keras.regularizers.L2(l2=weight)


def remove_some_samples(labels, model_config: LayerwiseModelConfig, mask):
    """Returns a random label mask removing some labeled and unlabeled samples."""
    if (
        model_config.max_prob_remove_unlabeled <= 0.0
        and model_config.max_prob_remove_labeled <= 0.0
    ):
        return mask

    if model_config.max_prob_remove_unlabeled > 0.0:
        # Dropping samples with a random probability between 0 and
        # `max_prob_remove_unlabeled`.
        prob = get_remove_probability(model_config.max_prob_remove_unlabeled)
        # Removing unlabeled samples with probability `prob`
        new_mask = tf.cast(tf.math.equal(labels, model_config.num_labels), tf.float32)
        masked_uniform = new_mask * tf.random.uniform(shape=new_mask.shape)
        mask_unlabeled = tf.cast(masked_uniform > 1 - prob, tf.float32)
    else:
        mask_unlabeled = tf.zeros_like(labels, dtype=tf.float32)

    if model_config.max_prob_remove_labeled > 0.0:
        # Dropping samples with a random probability between 0 and
        # `max_prob_remove_labeled`.
        prob = get_remove_probability(model_config.max_prob_remove_labeled)
        # Removing labeled samples with probability `prob`
        new_mask = tf.cast(
            tf.math.not_equal(labels, model_config.num_labels), tf.float32
        )
        masked_uniform = new_mask * tf.random.uniform(shape=new_mask.shape)
        mask_labeled = tf.cast(masked_uniform > 1 - prob, tf.float32)
    else:
        mask_labeled = tf.zeros_like(labels, dtype=tf.float32)

    # Boolean "or" equivalent for 3 masks (1 indicates a missing value).
    if mask is None:
        return tf.clip_by_value(mask_labeled + mask_unlabeled, 0.0, 1.0)
    else:
        return tf.clip_by_value(mask_labeled + mask_unlabeled + mask, 0.0, 1.0)


# ------------------------------------------------------------
#   Layer weight generators
# ------------------------------------------------------------


class Generator:
    """Generic generator."""

    def __init__(self, name: str, model_config: LayerwiseModelConfig):
        self.name = name
        self.model_config = model_config
        self.num_weight_blocks = None
        self.weight_block_size = None
        self.feature_extractor = None
        self.feature_extractor_class = None
        self.transformer_io = None
        self.transformer = None

    def _setup(self):
        raise NotImplementedError

    def set_weight_params(self, num_weight_blocks, weight_block_size):
        self.num_weight_blocks = num_weight_blocks
        self.weight_block_size = weight_block_size

    def set_feature_extractor_class(self, feature_extractor_class):
        self.feature_extractor_class = feature_extractor_class

    def _make_feature_extractor(self):
        if self.feature_extractor_class is not None:
            self.feature_extractor = self.feature_extractor_class(
                name="feature_extractor"
            )

    def _features(self, input_tensor, shared_features=None, enable_fe_dropout=False):
        """Returns full feature vector (per-layer and shared if specified)."""
        if self.feature_extractor is not None:
            # feature_extractor -> Activation Feature Extractor
            # shared_features   -> Image Feature Extractor
            features = self.feature_extractor(input_tensor)
            if enable_fe_dropout and self.model_config.fe_dropout > 0.0:
                dropout = tf.layers.Dropout(rate=self.model_config.fe_dropout)
                features = dropout(features, training=True)
        else:
            features = None

        if shared_features is not None:
            if features is not None:
                return tf.concat([features, shared_features], axis=-1)
            return shared_features

        if features is None:
            raise RuntimeError(
                "Layerwise model should have at least one of "
                "per-layer and shared feature extractors."
            )
        return features


class JointGenerator(Generator):
    """Model that feeds the Encoder/Decoder concatenated samples and weights."""

    def _pad_features(self, features):
        """Pads features to fit the embedding size."""
        feature_size = int(features.shape[1])
        embedding_dim = self.transformer_io.embedding_dim
        input_embedding_size = feature_size + embedding_dim

        if self.weight_block_size is None:
            raise RuntimeError("weight_block_size must be set before calling _pad_features()")

        embedding_size = max(
            embedding_dim + self.weight_block_size, input_embedding_size
        )
        if embedding_size == input_embedding_size:
            return features, embedding_size
        else:
            pad_size = embedding_size - input_embedding_size
            """
            Paddings specification for `tf.pad`:
               [
                   [0, 0],       # Dimension 0 (batch dimension):
                                 #   - pad_before = 0
                                 #   - pad_after  = 0
                                 #   → no padding is applied to the batch dimension

                   [0, pad_size] # Dimension 1 (feature dimension):
                                 #   - pad_before = 0
                                 #   - pad_after  = pad_size
                                 #   → append `pad_size` zeros to the right (end) of the feature vector
               ]

            Example:
               features = [[1, 2, 3],
                           [4, 5, 6]]
                                ↓
               paddings = [[1, 1],  # pad 1 sample before and after the batch
                           [0, 2]]  # pad 2 zeros at the end of the feature dimension
                                ↓
               tf.pad(features, paddings, mode="CONSTANT")
                                ↓
               [[0, 0, 0, 0, 0],   # padded batch (before)
                [1, 2, 3, 0, 0],   # original sample 0
                [4, 5, 6, 0, 0],   # original sample 1
                [0, 0, 0, 0, 0]]   # padded batch (after)
            """
            paddings = tf.constant([[0, 0], [0, pad_size]])
            return tf.pad(features, paddings, "CONSTANT"), embedding_size

    def get_transformer_params(self, embedding_size: int):
        """Returns Transformer parameters."""
        num_heads = self.model_config.heads

        def get_size(frac):
            if frac <= 3.0:
                dim = int(embedding_size * frac)
                if dim % num_heads > 0:
                    # Making sure that the Transformer input dimension is divisible by
                    # the number of heads.
                    dim = math.ceil(float(dim) / num_heads) * num_heads
                return dim
            else:
                return int(frac)

        attn_act_fn = common_ht.get_transformer_activation(self.model_config)

        return transformer.TransformerParams(
            query_key_dim=get_size(self.model_config.query_key_dim_frac), # D_qk
            internal_dim=get_size(self.model_config.internal_dim_frac), # Used by `PWFeedForward`
            value_dim=get_size(self.model_config.value_dim_frac), # D_v
            num_layers=self.model_config.num_layers,
            mha_output_dim=embedding_size,
            heads=num_heads,
            dropout_rate=self.model_config.dropout_rate,
            attention_activation_fn=attn_act_fn,
            activation_fn=util.nonlinearity(self.model_config.transformer_nonlinearity),
        )

    def _setup(self):
        with tf.variable_scope(self.name + "_setup"):
            self._make_feature_extractor()

            assert self.num_weight_blocks
            assert self.weight_block_size
            self.transformer_io = util.JointTransformerIO(
                num_labels=self.model_config.num_labels,
                num_weights=self.num_weight_blocks,
                embedding_dim=self.model_config.embedding_dim,
                weight_block_size=self.weight_block_size,
            )

    def generate_weights(
        self,
        input_tensor,
        labels,
        mask=None,
        shared_features=None,
        enable_fe_dropout=False,
    ):
        """Generates weights from the inputs."""
        if self.transformer_io is None:
            self._setup()

        with tf.variable_scope(f"builder_{self.name}"):
            features = self._features(
                input_tensor, shared_features, enable_fe_dropout=enable_fe_dropout
            )
            with tf.variable_scope("feature_padding"):
                features, transformer_embedding_size = self._pad_features(features)

            with tf.variable_scope("transformer"):
                if self.transformer is None:
                    if self.model_config.use_decoder:
                        model = transformer.EncoderDecoderModel
                    else:
                        model = transformer.EncoderModel
                    self.transformer = model(
                        self.get_transformer_params(transformer_embedding_size),
                        skip_last_nonlinearity=self.model_config.skip_last_nonlinearity,
                        name="transformer",
                    )
                if mask is not None:
                    mask = self.transformer_io.extend_label_mask(mask)
                transformer_input = self.transformer_io.encode_samples(features, labels)
                transformer_output = self.transformer(transformer_input, mask=mask)
                return self.transformer_io.decode_weights(transformer_output)


class SeparateGenerator(Generator):
    """Model that feeds samples to Encoder and weights to Decoder."""

    def get_encoder_params(self, embedding_size: int):
        """Returns Transformer parameters."""
        def get_size(frac):
            if frac <= 3.0:
                return int(embedding_size * frac)
            else:
                return int(frac)

        attn_act_fn = common_ht.get_transformer_activation(self.model_config)

        return transformer.TransformerParams(
            query_key_dim=get_size(self.model_config.query_key_dim_frac),
            internal_dim=get_size(self.model_config.internal_dim_frac),
            value_dim=get_size(self.model_config.value_dim_frac),
            num_layers=self.model_config.num_layers,
            mha_output_dim=embedding_size,
            heads=self.model_config.heads,
            dropout_rate=self.model_config.dropout_rate,
            attention_activation_fn=attn_act_fn,
            activation_fn=util.nonlinearity(self.model_config.transformer_nonlinearity),
        )

    def get_decoder_params(self, embedding_size: int):
        """Returns Transformer parameters."""
        def get_size(frac):
            if frac <= 3.0:
                return int(embedding_size * frac)
            else:
                return int(frac)

        attn_act_fn = common_ht.get_transformer_activation(self.model_config)

        return transformer.TransformerParams(
            query_key_dim=get_size(self.model_config.query_key_dim_frac),
            internal_dim=get_size(self.model_config.internal_dim_frac),
            value_dim=get_size(self.model_config.value_dim_frac),
            num_layers=self.model_config.num_layers,
            mha_output_dim=embedding_size,
            heads=self.model_config.heads,
            dropout_rate=self.model_config.dropout_rate,
            attention_activation_fn=attn_act_fn,
            activation_fn=util.nonlinearity(self.model_config.transformer_nonlinearity),
        )

    def _setup(self):
        with tf.variable_scope(self.name + "_setup"):
            self._make_feature_extractor()

            assert self.num_weight_blocks
            assert self.weight_block_size
            self.transformer_io = util.SeparateTransformerIO(
                num_labels=self.model_config.num_labels,
                num_weights=self.num_weight_blocks,
                embedding_dim=self.model_config.embedding_dim,
                weight_block_size=self.weight_block_size,
            )

    def generate_weights(self, input_tensor, labels, mask=None, shared_features=None, enable_fe_dropout=False):
        """Generates weights from the inputs."""
        del mask
        if self.transformer_io is None:
            self._setup()
        if self.model_config.max_prob_remove_unlabeled > 0:
            raise ValueError(
                "Removing unlabeled samples is not currently supported "
                'in the "separate" weight generator.'
            )

        assert self.weight_block_size is not None, \
        "weight_block_size must be set before calling _pad_features()"

        with tf.variable_scope(f"builder_{self.name}"):
            features = self._features(input_tensor, shared_features, enable_fe_dropout)
            weight_dim = self.transformer_io.weight_embedding_dim + self.weight_block_size
            sample_dim = self.transformer_io.embedding_dim
            sample_dim += int(features.shape[1])

            with tf.variable_scope("transformer"):
                if self.transformer is None:
                    self.transformer = transformer.SeparateEncoderDecoderModel(
                        encoder_params=self.get_encoder_params(sample_dim),
                        decoder_params=self.get_decoder_params(weight_dim),
                        skip_last_nonlinearity=self.model_config.skip_last_nonlinearity,
                        name="transformer",
                    )
                decoded = self.transformer(
                    self.transformer_io.encode_samples(features, labels),
                    self.transformer_io.encode_weights(),
                )
                return self.transformer_io.decode_weights(decoded)


# ------------------------------------------------------------
#   Base CNN Layer class
# ------------------------------------------------------------


class BaseCNNLayer(tf.Module):
    """Base CNN layer used in our models."""

    def __init__(
        self,
        name: str,
        model_config: LayerwiseModelConfig,
        head_builder=None,
        var_reg_weight=None,
    ):
        super().__init__(name=name)

        self.model_config = model_config
        self.num_labels = model_config.num_labels

        if var_reg_weight is None:
            var_reg_weight = model_config.var_reg_weight
        self.var_reg_weight = var_reg_weight

        self.feature_extractor = None
        self.head = None

        if head_builder is not None and self.model_config.train_heads:
            self.head = head_builder(
                name="head_" + self.name, model_config=self.model_config
            )

        if model_config.generator == "joint":
            self.generator = JointGenerator(
                name=self.name + "_generator", model_config=self.model_config
            )
        elif model_config.generator == "separate":
            self.generator = SeparateGenerator(
                name=self.name + "_generator", model_config=self.model_config
            )

        self.getter_dict = {}
        self.initialized = False

    def _setup(self, inputs):
        """Input-dependent layer setup."""
        raise NotImplementedError

    def __call__(self, inputs, training=True):
        raise NotImplementedError

    def _var_getter(self, offsets, weights, shape, name):
        raise NotImplementedError

    def create(
        self,
        input_tensor,
        labels,
        mask=None,
        shared_features=None,
        enable_fe_dropout=False,
        generate_weights=False,
    ):
        """Creates a layer using a feature extractor and a Transformer."""
        if not self.initialized:
            self._setup(input_tensor)
            self.generate_weights = generate_weights
            self.initialized = True

        if not self.generate_weights:
            return None

        return self.generator.generate_weights(
            input_tensor=input_tensor,
            labels=labels,
            mask=mask,
            shared_features=shared_features,
            enable_fe_dropout=enable_fe_dropout,
        )

    def apply(
        self,
        input_tensor,
        weight_blocks,
        *args,
        evaluation=False,
        separate_bn_variables=False,
        **kwargs,
    ):
        """Applies created layer to the input tensor.

        Attributes:
           Position parameters → *args → Keyword-only parameters → **kwargs
        """
        assert self.initialized

        variable_getter = functools.partial(
            util.var_getter_wrapper,
            _cnn_var_getter=self._var_getter,
            _weights=weight_blocks,
            _getter_dict=self.getter_dict,
            _add_trainable_weights=self.model_config.add_trainable_weights,
            _var_reg_weight=self.var_reg_weight,
            _kernel_regularizer=get_l2_regularizer(self.model_config.l2_reg_weight),
            _evaluation=evaluation,
            _separate_bn=separate_bn_variables,
        )
        with tf.variable_scope(
            self.name, reuse=tf.AUTO_REUSE, custom_getter=variable_getter
        ):
            # Equal to call `self.__call__(...)`
            return self(input_tensor, *args, **kwargs)


# ------------------------------------------------------------
#   Layer implementations: convolutional and logits layers
# ------------------------------------------------------------


class ConvLayer(BaseCNNLayer):
    """Conv Layer of the CNN."""

    def __init__(
        self,
        name,
        model_config,
        output_dim=None,
        kernel_size=None,
        num_features=None,
        feature_layers=0,
        padding="valid",
        generate_bias=None,
        generate_bn=None,
        stride=None,
        act_fn=None,
        act_after_bn=False,
        maxpool_size=None,
        head_builder=None,
    ):
        super().__init__(
            name=name,
            model_config=model_config,
            head_builder=head_builder,
        )

        if generate_bn is None:
            generate_bn = model_config.generate_bn
        if generate_bias is None:
            generate_bias = model_config.generate_bias
        if act_fn is None:
            act_fn = common_ht.get_cnn_activation(model_config)
        if kernel_size is None:
            kernel_size = model_config.kernel_size
        if num_features is None:
            num_features = model_config.num_features
        if output_dim is None:
            output_dim = model_config.default_num_channels
        if stride is None:
            stride = model_config.stride

        self.generate_bn = generate_bn
        self.generate_bias = generate_bias
        self.act_fn = act_fn
        self.kernel_size = kernel_size
        self.num_features = num_features

        self.feature_layers = feature_layers
        if self.feature_layers < 1:
            self.feature_layers = self.model_config.feature_layers

        self.padding = padding
        self.input_dim = -1
        self.output_dim = output_dim
        self.stride = stride
        self.maxpool_size = maxpool_size
        self.initialized = False
        self.act_after_bn = act_after_bn

    def _compute_weight_sizes(self, weight_alloc):
        """Computes the number of weight blocks, their size, axis to stack, etc."""
        assert self.input_dim > 0
        if weight_alloc == common_ht.WeightAllocation.SPATIAL:
            if self.generate_bn:
                raise ValueError(
                    "BN weight generation is not currently supported for "
                    "the spatial weight allocation."
                )
            if self.generate_bias:
                raise ValueError(
                    "Bias generation is not currently supported for "
                    "the spatial weight allocation."
                )

            self.num_blocks = int(self.kernel_size**2)
            self.block_size = self.input_dim * self.output_dim
            self.conv_weight_size = self.block_size

            self.stack_axis = 0
        elif weight_alloc == common_ht.WeightAllocation.OUTPUT_CHANNEL:
            self.num_blocks = self.output_dim
            self.block_size = self.input_dim * int(self.kernel_size**2)
            self.conv_weight_size = self.block_size

            if self.generate_bias:
                self.block_size += 1 # + 1 bias
            if self.generate_bn:
                self.block_size += 2 # + gamma + beta

            self.stack_axis = -1
        else:
            raise ValueError("Unknown WeightAllocation value.")

    def _setup(self, inputs):
        """Input-specific setup."""
        self.input_dim = int(inputs.shape[-1])
        if self.num_features is None:
            self.num_features = max(self.output_dim, self.input_dim)
        self._compute_weight_sizes(self.model_config.weight_allocation)
        feature_extractor_class = functools.partial(
            feature_extractors.SimpleConvFeatureExtractor,
            input_size=int(inputs.shape[1]),
            feature_layers=self.feature_layers,
            feature_dim=self.num_features,
            kernel_size=self.kernel_size,
        )
        self.generator.set_weight_params(
            num_weight_blocks=self.num_blocks, weight_block_size=self.block_size
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

    def __call__(self, inputs, training=True):
        # Layers should be created in `__call__` to properly build weights from
        # Transformer outputs for both training and evaluation.
        self.conv = tf.layers.Conv2D(
            filters=self.output_dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=self.padding,
            use_bias=self.generate_bias,
            name="conv",
        )
        self.bn = tf.layers.BatchNormalization()
        output = self.conv(inputs)
        if not self.act_after_bn:
            output = self.act_fn(output)
        if self.maxpool_size is not None:
            maxpool = tf.keras.layers.MaxPool2D(
                pool_size=(self.maxpool_size, self.maxpool_size),
                strides=(self.maxpool_size, self.maxpool_size),
                padding="valid",
            )
            output = maxpool(output)
        
        """
        # Batch normalization is always in the training mode.
        #
        # ------------------------------------------------------------------
        # In a standard CNN:
        #   - During training, training=True  -> use batch mean/variance
        #   - During testing,  training=False -> use global mean/variance (moving averages) from training
        #   Assumes train/test data come from the same distribution.
        #
        # In this hypertransformer + few-shot setting:
        #   - Each forward pass is a new task (e.g., "cat vs dog", then "plane vs ship")
        #   - CNN weights are generated per task, not fixed
        #   - There is no stable global feature distribution
        #
        # Therefore, BatchNorm must always use batch statistics (training=True)
        # to normalize features based on the current task's data.
        """
        output = self.bn(output, training=True)
        if self.act_after_bn:
            output = self.act_fn(output)
        return output

    def _var_getter(self, offsets, weights, shape, name):
        if name.endswith("/kernel"):
            if self.generate_weights:
                """Take the kernel part from each w
                w = [
                    conv_kernel_flat...,   # conv_weight_size
                    bias,                  # 1 bias
                    gamma,                 # 1 BN
                    beta                   # 1 BN
                ]
                    |
                    V
                ws = [
                    [k1_0, k1_1, ..., k1_N],
                    [k2_0, k2_1, ..., k2_N],
                    ...
                ]
                """
                ws = [w[: self.conv_weight_size] for w in weights]
                kernel = tf.stack(ws, axis=self.stack_axis)
                # [k_H, k_W, C_in, C_out]
                kernel = tf.reshape(
                    kernel,
                    (
                        self.kernel_size,
                        self.kernel_size,
                        self.input_dim,
                        self.output_dim,
                    ),
                )
                return kernel

        if name.endswith("/bias"):
            assert self.generate_bias
            if self.generate_weights:
                return tf.stack([w[self.conv_weight_size] for w in weights], axis=0)

        if self.generate_bn:
            offset = self.conv_weight_size + int(self.generate_bias)
            if name.endswith("/gamma"):
                tensor = tf.stack([w[offset] for w in weights])
                return GAMMA_BIAS + tensor * GAMMA_SCALE
            elif name.endswith("/beta"):
                tensor = tf.stack([w[offset + 1] for w in weights])
                return BETA_BIAS + tensor * BETA_SCALE
            else:
                return None

        return None


class LogitsLayer(BaseCNNLayer):
    """Logits layer of the CNN.

       [batch, H, W, C]
            ↓
       [batch, C]
            ↓
       [batch, num_labels]
    """

    def __init__(
        self,
        name: str,
        model_config: LayerwiseModelConfig,
        num_features: Optional[int] = None,
        fe_kernel_size=3,
        head_builder=None,
    ):
        super().__init__(
            name=name,
            model_config=model_config,
            head_builder=head_builder,
            # We generally do not want to regularize the last logits layer.
            var_reg_weight=0.0,
        )

        self.dropout = tf.layers.Dropout(rate=model_config.cnn_dropout_rate)

        if num_features is None:
            num_features = model_config.num_features
        self.num_features = num_features

        self.fe_kernel_size = fe_kernel_size
        self.input_dim = -1
        self.initialized = False

    def _setup(self, inputs):
        """Input-specific setup."""
        self.input_dim = int(inputs.shape[-1])

        if self.num_features is None:
            self.num_features = self.input_dim
        feature_extractor_class = functools.partial(
            feature_extractors.SimpleConvFeatureExtractor,
            feature_layers=1,
            input_size=int(inputs.shape[1]),
            feature_dim=self.num_features,
            nonlinear_feature=self.model_config.nonlinear_feature,
            kernel_size=self.fe_kernel_size,
        )

        self.generator.set_weight_params(
            num_weight_blocks=self.model_config.num_labels,
            weight_block_size=self.input_dim + 1, # + 1 bias
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

    def __call__(self, tensor, training=True):
        self.fc = tf.layers.Dense(units=self.model_config.num_labels, name="fc")
        # Global Average Pooling
        # [batch, height, width, channels]
        #       ↓
        # [batch, channels]
        tensor = tf.reduce_mean(tensor, axis=[1, 2])
        dropout_tensor = self.dropout(tensor, training=training)
        # [batch, channels] → [batch, num_labels]
        return self.fc(dropout_tensor)

    def _var_getter(self, offsets, weights, shape, name):
        if weights is None:
            return None

        assert self.generator.weight_block_size is not None, \
        "weight_block_size must be set before calling _pad_features"

        if name.endswith("/kernel"):
            n = self.generator.weight_block_size - 1
            ws = [w[:n] for w in weights]
            return tf.stack(ws, axis=-1)
        if name.endswith("/bias"):
            n = self.generator.weight_block_size - 1
            ws = [w[n] for w in weights]
            output = tf.stack(ws, axis=-1)
            return output

        return None


class FlattenLogitsLayer(LogitsLayer):
    """Logits layer of the CNN that flattens its input (instead of averaging).

       [batch, H, W, C]
            ↓
       [batch, H*W*C]
            ↓
       [batch, num_labels]
    """

    def _setup(self, inputs):
        """Input-specific setup."""
        self.input_dim = int(inputs.shape[-1])

        width, height = int(inputs.shape[1]), int(inputs.shape[2])

        if self.num_features is None:
            self.num_features = self.input_dim
        if self.model_config.logits_feature_extractor in ["", "default", "mix"]:
            feature_extractor_class = functools.partial(
                feature_extractors.SimpleConvFeatureExtractor,
                feature_layers=1,
                feature_dim=self.num_features,
                input_size=int(inputs.shape[1]),
                nonlinear_feature=self.model_config.nonlinear_feature,
                kernel_size=self.fe_kernel_size,
            )
        elif self.model_config.logits_feature_extractor == "passthrough":
            feature_extractor_class = feature_extractors.PassthroughFeatureExtractor
        else:
            raise AssertionError("Unexpected `logits_feature_extractor` value.")

        if self.model_config.logits_feature_extractor == "mix":
            feature_extractor_class = functools.partial(
                feature_extractors.PassthroughFeatureExtractor,
                wrap_class=feature_extractor_class,
            )

        self.generator.set_weight_params(
            num_weight_blocks=self.model_config.num_labels,
            weight_block_size=self.input_dim * width * height + 1, # + 1 bias
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

    def __call__(self, tensor, training=True):
        # [batch, H, W, C] → [batch, H*W*C]
        flatten = tf.layers.Flatten()

        # [batch, H*W*C] → [batch, num_labels]
        self.fc = tf.layers.Dense(units=self.model_config.num_labels, name="fc")
        dropout_tensor = self.dropout(flatten(tensor), training=training)
        return self.fc(dropout_tensor)


# ------------------------------------------------------------
#   Layerwise model class
# ------------------------------------------------------------


class LayerwiseModel(common_ht.Model):
    """Model specification including layer builders."""

    def __init__(self, layers: "list[ConvLayer | BaseCNNLayer]", model_config: LayerwiseModelConfig):
        super().__init__()

        self.layers = layers
        self.shared_feature_extractor = feature_extractors.get_shared_feature_extractor(
            model_config
        )
        self.separate_bn_variables = model_config.separate_bn_vars

        #   = 0 : freeze all CNN layers (no backpropagation through CNN)
        #   > 0 : train the first N CNN layers
        #   < 0 : train the last |N| CNN layers
        self._number_of_trained_cnn_layers = model_config.number_of_trained_cnn_layers
        self.shared_fe_dropout = model_config.shared_fe_dropout
        self.fe_dropout = model_config.fe_dropout
        self.model_config = model_config

    def train(
        self,
        inputs,
        labels,
        mask=None,
        mask_random_samples=False,
        enable_fe_dropout=False,
        only_shared_feature=False,
    ):
        """Builds an entire CNN model using train inputs."""
        all_weight_blocks = []
        all_head_blocks = {}

        shared_features = None
        if self.shared_feature_extractor is not None:
            shared_features = self.shared_feature_extractor(inputs)
            if enable_fe_dropout and self.shared_fe_dropout > 0.0:
                dropout = tf.layers.Dropout(rate=self.shared_fe_dropout)
                shared_features = dropout(shared_features, training=True)

        if only_shared_feature:
            return GeneratedWeights(
                weight_blocks=[],
                head_weight_blocks={},
                shared_features=shared_features,
            )

        if mask_random_samples:
            mask = remove_some_samples(labels, self.model_config, mask)

        num_trained_layers = abs(self._number_of_trained_cnn_layers)
        # Last layer is always a LogitsLayer and we always generate it.
        num_generated_layers = len(self.layers) - num_trained_layers - 1
        if num_generated_layers < 0:
            raise ValueError(
                "num_trained_layers should be smaller that the total "
                "number of conv layers."
            )

        is_first_trained = self._number_of_trained_cnn_layers >= 0
        if is_first_trained:
            generate_weights_per_layers = (
                [False] * num_trained_layers + [True] * num_generated_layers + [True]
            )
        else:
            generate_weights_per_layers = (
                [True] * num_generated_layers + [False] * num_trained_layers + [True]
            )

        for layer, generate_weights in zip(self.layers, generate_weights_per_layers):
            with tf.variable_scope("cnn_builder"):
                weight_blocks = layer.create(
                    input_tensor=inputs,
                    labels=labels,
                    mask=mask,
                    shared_features=shared_features,
                    enable_fe_dropout=enable_fe_dropout,
                    generate_weights=generate_weights,
                )
                all_weight_blocks.append(weight_blocks)
            with tf.variable_scope("cnn"):
                inputs = layer.apply(
                    inputs,
                    weight_blocks=weight_blocks,
                    training=True,
                    separate_bn_variables=self.separate_bn_variables,
                )

            if layer.head is not None:
                with tf.variable_scope("cnn_builder_heads"):
                    head_blocks = layer.head.create(
                        input_tensor=inputs,
                        labels=labels,
                        mask=mask,
                        shared_features=shared_features,
                        enable_fe_dropout=enable_fe_dropout,
                    )
                    all_head_blocks[layer.name] = head_blocks

        return GeneratedWeights(
            weight_blocks=all_weight_blocks,
            head_weight_blocks=all_head_blocks,
            shared_features=shared_features,
        )

    def evaluate(
        self,
        inputs,
        weight_blocks=None,
        training=True,
    ):
        """Passes input tensors through a built CNN model."""
        if weight_blocks is None:
            raise ValueError("weight_blocks must be provided")

        self.layer_outputs = {}
        with tf.variable_scope("cnn"):
            for layer, layer_blocks in zip(self.layers, weight_blocks.weight_blocks):
                inputs = layer.apply(
                    inputs,
                    weight_blocks=layer_blocks,
                    training=training,
                    evaluation=True,
                    separate_bn_variables=self.separate_bn_variables,
                )

                head = None
                if layer.head is not None:
                    head = layer.head.apply(
                        inputs,
                        weight_blocks=weight_blocks.head_weight_blocks[layer.name],
                        training=training,
                        evaluation=True,
                        separate_bn_variables=self.separate_bn_variables,
                    )
                self.layer_outputs[layer.name] = (inputs, head)

        return inputs
