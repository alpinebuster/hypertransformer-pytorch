"""Core implementations of "layerwise" models."""

import dataclasses
import functools
import math

from typing import Callable, Optional, Union, Sequence, \
    Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypertransformer.core import common_ht
from hypertransformer.core.common_ht import LayerwiseModelConfig
from hypertransformer.core import feature_extractors
from hypertransformer.core import transformer
from hypertransformer.core import util

GAMMA_SCALE = 1.0
BETA_SCALE = 1.0
GAMMA_BIAS = 0.0
BETA_BIAS = 0.0

models: dict[str, Callable[..., "LayerwiseModel"]] = {}

HeadBuilder = Callable[..., "BaseCNNLayer"]


@dataclasses.dataclass
class GeneratedWeights:
    weight_blocks: list[list[torch.Tensor]]
    head_weight_blocks: dict[str, list[torch.Tensor]]
    shared_features: Optional[torch.Tensor] = None


def build_model(name: str, model_config: common_ht.LayerwiseModelConfig) -> "LayerwiseModel":
    model_fn = models[name]
    return model_fn(model_config=model_config)


def get_remove_probability(max_probability: float) -> torch.Tensor:
    """Returns a random probability between 0 and `max_probability`.

    e.g.
       >>> torch.rand(size=()) 
           tensor(0.6942)
    """
    return torch.rand(size=()) * max_probability


def remove_some_samples(
    labels: torch.Tensor,
    model_config: "LayerwiseModelConfig",
    mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Returns a random label mask removing some labeled and unlabeled samples."""
    if (
        model_config.max_prob_remove_unlabeled <= 0.0
        and model_config.max_prob_remove_labeled <= 0.0
    ):
        return mask

    # Unlabeled samples
    if model_config.max_prob_remove_unlabeled > 0.0:
        # Dropping samples with a random probability between 0 and
        # `max_prob_remove_unlabeled`.
        prob = get_remove_probability(model_config.max_prob_remove_unlabeled)
        # Removing unlabeled samples with probability `prob`
        # A label with a value of `num_labels` indicates no label
        new_mask = (labels == model_config.num_labels).float()
        masked_uniform = new_mask * torch.rand_like(new_mask)
        mask_unlabeled = (masked_uniform > 1-prob).float()
    else:
        mask_unlabeled = torch.zeros_like(labels, dtype=torch.float32)

    # Labeled samples
    if model_config.max_prob_remove_labeled > 0.0:
        # Dropping samples with a random probability between 0 and
        # `max_prob_remove_labeled`.
        prob = get_remove_probability(model_config.max_prob_remove_labeled)
        # Removing labeled samples with probability `prob`
        new_mask =  (labels != model_config.num_labels).float()
        masked_uniform = new_mask * torch.rand_like(new_mask)
        mask_labeled = (masked_uniform > 1-prob).float()
    else:
        mask_labeled = torch.zeros_like(labels, dtype=torch.float32)

    # Combine masks
    # Boolean "or" equivalent for 3 masks (1 indicates a missing value).
    if mask is None:
        return torch.clamp(mask_labeled + mask_unlabeled, 0.0, 1.0)
    else:
        return torch.clamp(mask_labeled + mask_unlabeled + mask, 0.0, 1.0)


# ------------------------------------------------------------
#   Layer weight generators
# ------------------------------------------------------------


class _Generator:
    """Generic generator."""

    def __init__(self, name: str, model_config: "LayerwiseModelConfig"):
        self.name = name
        self.model_config = model_config
        self.num_weight_blocks: Optional[int] = None
        self.weight_block_size: Optional[int] = None
        self.feature_extractor: Optional[nn.Module] = None
        self.feature_extractor_class: Optional[Callable[..., nn.Module]] = None
        self.transformer_io: Optional[Union[
            util.JointTransformerIO,
            util.SeparateTransformerIO,
        ]] = None
        self.transformer: Optional[Union[
            transformer.EncoderDecoderModel,
            transformer.EncoderModel,
            transformer.SeparateEncoderDecoderModel,
        ]] = None

    def _setup(self):
        raise NotImplementedError

    def set_weight_params(self, num_weight_blocks: int, weight_block_size: int):
        self.num_weight_blocks = num_weight_blocks
        self.weight_block_size = weight_block_size

    def set_feature_extractor_class(
        self,
        feature_extractor_class: Callable[..., nn.Module],
    ) -> None:
        self.feature_extractor_class = feature_extractor_class

    def _make_feature_extractor(self):
        if self.feature_extractor_class is not None:
            self.feature_extractor = self.feature_extractor_class(
                name="feature_extractor"
            )

    def _features(
        self,
        input_tensor: torch.Tensor,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False
    ) -> torch.Tensor:
        """Returns full feature vector (per-layer and shared if specified)."""
        if self.feature_extractor is not None:
            # feature_extractor -> Activation Feature Extractor
            # shared_features   -> Image Feature Extractor
            features = self.feature_extractor(input_tensor)
            if enable_fe_dropout and self.model_config.fe_dropout > 0.0:
                dropout = nn.Dropout(p=self.model_config.fe_dropout)
                features = dropout(features)
        else:
            features = None

        if shared_features is not None:
            if features is not None:
                return torch.concat([features, shared_features], dim=-1)
            return shared_features

        if features is None:
            raise RuntimeError(
                "Layerwise model should have at least one of "
                "per-layer and shared feature extractors."
            )
        return features


class JointGenerator(_Generator):
    """Model that feeds the Encoder/Decoder concatenated samples and weights."""

    def _pad_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pads features to fit the embedding size.

        [batch_size, feature_dim]
            ↓
        [batch_size, feature_dim + pad_size]
        """
        assert isinstance(self.transformer_io, util.JointTransformerIO)

        feature_size = int(features.shape[1])
        embedding_dim = self.transformer_io.embedding_dim
        input_embedding_size = feature_size + embedding_dim

        if self.weight_block_size is None:
            raise RuntimeError("weight_block_size must be set before calling _pad_features()")

        embedding_size = max(
            embedding_dim + self.weight_block_size,
            input_embedding_size,
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
            padded_features = F.pad(features, (0, pad_size), mode='constant', value=0)
            return padded_features, embedding_size

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
        input_tensor: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False,
    ) -> list[torch.Tensor]:
        """Generates weights from the inputs."""
        if self.transformer_io is None:
            self._setup()

        assert isinstance(self.transformer_io, util.JointTransformerIO)

        features = self._features(
            input_tensor, shared_features, enable_fe_dropout=enable_fe_dropout
        )
        features, transformer_embedding_size = self._pad_features(features)

        if self.transformer is None:
            if self.model_config.use_decoder:
                model_class = transformer.EncoderDecoderModel
            else:
                model_class = transformer.EncoderModel
            self.transformer = model_class(
                self.get_transformer_params(transformer_embedding_size),
                skip_last_nonlinearity=self.model_config.skip_last_nonlinearity,
                name="transformer",
            )
        if mask is not None:
            mask = self.transformer_io.extend_label_mask(mask)
        transformer_input = self.transformer_io.encode_samples(features, labels)
        transformer_output = self.transformer(transformer_input, mask=mask)

        return self.transformer_io.decode_weights(transformer_output)


class SeparateGenerator(_Generator):
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
        self._make_feature_extractor()

        assert self.num_weight_blocks
        assert self.weight_block_size
        self.transformer_io = util.SeparateTransformerIO(
            num_labels=self.model_config.num_labels,
            num_weights=self.num_weight_blocks,
            embedding_dim=self.model_config.embedding_dim,
            weight_block_size=self.weight_block_size,
        )

    def generate_weights(
        self,
        input_tensor: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False,
    ) -> list[torch.Tensor]:
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
        assert isinstance(self.transformer_io, util.SeparateTransformerIO)

        features = self._features(input_tensor, shared_features, enable_fe_dropout)
        weight_dim = self.transformer_io.weight_embedding_dim + self.weight_block_size
        sample_dim = self.transformer_io.embedding_dim
        sample_dim += int(features.shape[1])

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


def _get_l2_regularizer(weight=None):
    if weight is None or weight == 0.0:
        return None
    return tf.keras.regularizers.L2(l2=weight)

class BaseCNNLayer(nn.Module):
    """Base CNN layer used in our models."""

    def __init__(
        self,
        name: str,
        model_config: LayerwiseModelConfig,
        head_builder: Optional[Callable[..., "BaseCNNLayer"]] = None,
        var_reg_weight: Optional[float] = None,
    ):
        super().__init__()

        self.name = name
        self.model_config = model_config
        self.num_labels = model_config.num_labels

        if var_reg_weight is None:
            var_reg_weight = model_config.var_reg_weight
        self.var_reg_weight = var_reg_weight

        self.feature_extractor = None
        self.head: Optional["BaseCNNLayer"] = None

        if head_builder is not None and self.model_config.train_heads:
            self.head = head_builder(
                name="head_" + self.name,
                model_config=self.model_config,
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

    def _setup(self, inputs: torch.Tensor):
        """Input-dependent layer setup."""
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def _var_getter(self, offsets: dict, weights: list[torch.Tensor], shape, name: str):
        """
        name = "conv/kernel"
                │
                ▼
        cnn_var_getter(name)
                │
                ├── Tensor ──┐
                │            ├─ (+ trainable residual)
                │            └─ (+ regularization)
                │
                └── None ───▶ fallback_fn(name)
                                │
                                └─ nn.Parameter
        """
        raise NotImplementedError

    def create(
        self,
        input_tensor: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False,
        generate_weights: bool = False,
    ) -> Optional[list[torch.Tensor]]:
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
        input_tensor: torch.Tensor,
        weight_blocks: list[torch.Tensor],
        *args,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
        **kwargs,
    ):
        """Applies created layer to the input tensor.

        Attributes:
           Position parameters → *args → Keyword-only parameters → **kwargs

        Child.apply(x)
        → self(x)
        → nn.Module.__call__(x)
        → Child.forward(x)
        """
        assert self.initialized, "Layer must be initialized with `self.create()` before apply"

        variable_getter = functools.partial(
            util.var_getter_wrapper,
            _cnn_var_getter=self._var_getter,
            _weights=weight_blocks,
            _getter_dict=self.getter_dict,
            _add_trainable_weights=self.model_config.add_trainable_weights,
            _var_reg_weight=self.var_reg_weight,
            _kernel_regularizer=_get_l2_regularizer(self.model_config.l2_reg_weight),
            _evaluation=evaluation,
            _separate_bn=separate_bn_variables,
        )

        # Equal to call `self.__call__(...) or self.forward(...)`
        return self(input_tensor, *args, **kwargs)


# ------------------------------------------------------------
#   Layer implementations: convolutional and logits layers
# ------------------------------------------------------------


class ConvLayer(BaseCNNLayer):
    """Conv Layer of the CNN."""

    def __init__(
        self,
        name,
        model_config: "LayerwiseModelConfig",
        output_dim=None,
        kernel_size: Optional[int] = None,
        num_features=None,
        feature_layers=0,
        padding: str ="valid",
        generate_bias: Optional[bool] = None,
        generate_bn: Optional[bool] = None,
        stride=None,
        act_fn=None,
        act_after_bn=False,
        maxpool_size=None,
        head_builder: Optional[Callable[..., "BaseCNNLayer"]] = None,
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

        assert kernel_size is not None
        self.kernel_size = kernel_size

        self.num_features = num_features

        self.feature_layers = feature_layers
        if self.feature_layers < 1:
            self.feature_layers = self.model_config.feature_layers

        # PyTorch 1.10+ supports "same" / "valid"
        assert padding in ("same", "valid")
        self.padding = padding

        self.input_dim = -1
        self.output_dim = output_dim
        self.stride = stride
        self.maxpool_size = maxpool_size
        self.initialized = False
        self.act_after_bn = act_after_bn

    def _compute_weight_sizes(self, weight_alloc: common_ht.WeightAllocation):
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

    def _setup(self, inputs: torch.Tensor) -> None:
        """Input-specific setup."""
        # inputs -> [NCHW]
        self.input_dim = int(inputs.shape[-1])
        if self.num_features is None:
            self.num_features = max(self.output_dim, self.input_dim)
        self._compute_weight_sizes(self.model_config.weight_allocation)
        feature_extractor_class = functools.partial(
            feature_extractors.SimpleConvFeatureExtractor,
            in_channels=inputs.shape[1],
            input_size=int(inputs.shape[2]),
            feature_layers=self.feature_layers,
            feature_dim=self.num_features,
            kernel_size=self.kernel_size,
        )
        self.generator.set_weight_params(
            num_weight_blocks=self.num_blocks, weight_block_size=self.block_size
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

        self.conv = nn.Conv2d(
            in_channels=inputs.shape[1],
            out_channels=self.output_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.generate_bias,
        )
        self.bn = nn.BatchNorm2d(self.output_dim)
        if self.maxpool_size is not None:
            self.maxpool = nn.MaxPool2d(
                kernel_size=self.maxpool_size,
                stride=self.maxpool_size,
                padding=0, # padding="valid"
            )
        else:
            self.maxpool = None

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Layers should be created in `__call__` to properly build weights from
        # Transformer outputs for both training and evaluation.
        output = self.conv(inputs)
        if not self.act_after_bn:
            output = self.act_fn(output)
        if self.maxpool_size is not None and self.maxpool is not None:
            output = self.maxpool(output)

        output = self.bn(output)
        if self.act_after_bn:
            output = self.act_fn(output)
        return output

    def _var_getter(self, offsets: dict, weights: list[torch.Tensor], shape, name: str):
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
        head_builder: Optional[Callable[..., "BaseCNNLayer"]] = None,
    ):
        super().__init__(
            name=name,
            model_config=model_config,
            head_builder=head_builder,
            # We generally do not want to regularize the last logits layer.
            var_reg_weight=0.0,
        )

        self.dropout = nn.Dropout(p=model_config.cnn_dropout_rate)

        if num_features is None:
            num_features = model_config.num_features
        self.num_features = num_features

        self.fe_kernel_size = fe_kernel_size
        self.input_dim = -1
        self.initialized = False

    def _setup(self, inputs: torch.Tensor) -> None:
        """Input-specific setup."""
        # inputs -> [NCHW]
        self.input_dim = int(inputs.shape[-1])

        if self.num_features is None:
            self.num_features = self.input_dim
        feature_extractor_class = functools.partial(
            feature_extractors.SimpleConvFeatureExtractor,
            in_channels=inputs.shape[1],
            feature_layers=1,
            input_size=int(inputs.shape[2]),
            feature_dim=self.num_features,
            nonlinear_feature=self.model_config.nonlinear_feature,
            kernel_size=self.fe_kernel_size,
        )

        self.generator.set_weight_params(
            num_weight_blocks=self.model_config.num_labels,
            weight_block_size=self.input_dim + 1, # + 1 bias
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(
            in_features=inputs.shape[1],
            out_features=self.model_config.num_labels,
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Global Average Pooling
        # [batch, channels, height, width] → [batch, channels, 1, 1]
        inputs = self.gap(inputs)
        # [batch, channels, 1, 1] → [batch, channels]
        inputs = inputs.flatten(start_dim=1)
        dropout_tensor = self.dropout(inputs)
        # [batch, channels] → [batch, num_labels]
        return self.fc(dropout_tensor)

    def _var_getter(self, offsets: dict, weights: list[torch.Tensor], shape, name: str):
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

    def _setup(self, inputs: torch.Tensor) -> None:
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
                name="feature_extractor",
                wrap_class=feature_extractor_class,
            )

        self.generator.set_weight_params(
            num_weight_blocks=self.model_config.num_labels,
            weight_block_size=self.input_dim * width * height + 1, # + 1 bias
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

        self.fc = nn.Linear(
            in_features=inputs.shape[1],
            out_features=self.model_config.num_labels,
        )

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        # [batch, C, H, W] → [batch, C*H*W]
        inputs = inputs.flatten(start_dim=1)
        dropout_tensor = self.dropout(inputs)
        # [batch, C*H*W] → [batch, num_labels]
        return self.fc(dropout_tensor)


# ------------------------------------------------------------
#   Layerwise model class
# ------------------------------------------------------------


class LayerwiseModel(common_ht.Model):
    """Model specification including layer builders."""

    def __init__(self, layers: "Sequence[Union[ConvLayer, BaseCNNLayer]]", model_config: LayerwiseModelConfig):
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
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_random_samples: bool = False,
        enable_fe_dropout: bool = False,
        only_shared_feature: bool = False,
    ) -> GeneratedWeights:
        """Builds an entire CNN model using train inputs (Support Set)."""
        all_weight_blocks = []
        all_head_blocks = {}

        shared_features = None
        if self.shared_feature_extractor is not None:
            shared_features = self.shared_feature_extractor(inputs)
            if enable_fe_dropout and self.shared_fe_dropout > 0.0:
                dropout = nn.Dropout(p=self.shared_fe_dropout)
                shared_features = dropout(shared_features)

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
            # cnn_builder
            weight_blocks = layer.create(
                input_tensor=inputs,
                labels=labels,
                mask=mask,
                shared_features=shared_features,
                enable_fe_dropout=enable_fe_dropout,
                generate_weights=generate_weights,
            )
            all_weight_blocks.append(weight_blocks)

            # cnn
            assert weight_blocks is not None
            inputs = layer.apply(
                inputs,
                weight_blocks=weight_blocks,
                training=True,
                separate_bn_variables=self.separate_bn_variables,
            )

            # cnn_builder_heads
            if layer.head is not None:
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
        inputs: torch.Tensor,
        weight_blocks: Optional["GeneratedWeights"] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Passes input tensors (Query Set) through a built CNN model."""
        if weight_blocks is None:
            raise ValueError("weight_blocks must be provided")

        self.layer_outputs = {}

        for layer, layer_blocks in zip(self.layers, weight_blocks.weight_blocks):
            # cnn
            inputs = layer.apply(
                inputs,
                weight_blocks=layer_blocks,
                training=training,
                evaluation=True,
                separate_bn_variables=self.separate_bn_variables,
            )

            # cnn_builder_heads
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
