"""Feature extractors used to generate Layerwise models."""

import functools

from typing import Optional
import typing_extensions
from absl import logging

import tensorflow.compat.v1 as tf # pyright: ignore[reportMissingImports] # pylint:disable=import-error

from hypertransformer.core.common_ht import LayerwiseModelConfig

Protocol = typing_extensions.Protocol


class FeatureExtractor(tf.Module):
    pass


class SimpleConvFeatureExtractor(FeatureExtractor):
    """Simple convolutional feature extractor.

    Extract and splice multi-layer features layer by layer, e.g.:
      Input ([B, H, W, C])
      ├─ Conv1 → GAP (Global Average Pooling) → f1 ([B, C])
      ├─ Conv2 → GAP → f2 ([B, C])
      ├─ Conv3 → GAP → f3 ([B, C])
      └─ concat(f1, f2, f3) ([B, [f1_1, ..., f1_C, f2_1, ..., f2_C, f3_1, ..., f3_C]])
    """

    def __init__(
        self,
        feature_layers: int,
        feature_dim: int, # Number of Conv2D filters
        name: str,
        nonlinear_feature=False,
        kernel_size=3,
        input_size: Optional[int] = None,
        padding="valid",
    ):
        super().__init__(name=name)

        self.feature_dim = feature_dim
        self.nonlinear_feature = nonlinear_feature
        self.convs = []
        self.kernel_size = kernel_size
        def_stride = 2

        assert input_size
        if input_size < kernel_size:
            self.kernel_size = input_size
            def_stride = 1

        assert feature_dim > 0
        for idx, layer in enumerate(range(feature_layers)):
            # The first `feature_layers - 1` convolutional layers use a stride of 2 to progressively reduce the spatial dimensions,
            # while the final layer uses a stride of `1` to extract features without additional downsampling.
            stride = def_stride if idx < feature_layers - 1 else 1
            self.convs.append(
                tf.layers.Conv2D(
                    filters=feature_dim,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(stride, stride),
                    padding=padding,
                    activation=None,
                    name=f"layer_{layer + 1}",
                )
            )

    def __call__(self, input_tensor):
        if not self.convs:
            return None

        with tf.variable_scope(None, default_name=self.name):
            tensor = input_tensor
            outputs = []

            for conv in self.convs:
                if int(tensor.shape[1]) < self.kernel_size:
                    logging.info(
                        f"{self.__class__.__name__} cannot apply Conv2d at layer {len(outputs)+1}: "
                        f"spatial size {(tensor.shape[-2], tensor.shape[-1])} "
                        f"is smaller than kernel_size={self.kernel_size}. "
                        f"Check input_size, kernel_size, stride, or padding."
                    )
                    break

                tensor = conv(tensor)
                feature = tensor

                # While the output is not employing nonlinearity, layer-to-layer
                # transformations use it.
                tensor = tf.nn.relu(tensor)
                if self.nonlinear_feature:
                    feature = tensor
                outputs.append(feature)

            outputs = [tf.reduce_mean(tensor, axis=(1, 2)) for tensor in outputs]
            return tf.concat(outputs, axis=-1)


class SharedMultilayerFeatureExtractor(FeatureExtractor):
    """Simple shared convolutional feature extractor.

    Just like a standard CNN, it only outputs the last layer of features, e.g.:
      Input ([B, H, W, C_in])
        ↓
      Conv1 ([B, H/2^(1-1), W/2^(1-1), feature_dim])
        ↓
      Conv2 ([B, H/2^(2-1), W/2^(2-1), feature_dim])
        ↓
      Conv3 ([B, H/2^(3-1), W/2^(3-1), feature_dim])
        ↓
      GAP ([B, H', W', feature_dim] -> [B, feature_dim])
        ↓
      Features ([B, feature_dim])
    """

    def __init__(
        self,
        feature_layers: int,
        feature_dim: int, # Number of Conv2D filters
        name: str,
        kernel_size: int = 3,
        padding: str = "valid",
        use_bn: bool = False,
    ):
        super().__init__(name=name)

        self.feature_dim = feature_dim
        self.convs = []
        self.bns = []
        self.kernel_size = kernel_size

        assert feature_dim > 0
        for idx, layer in enumerate(range(feature_layers)):
            # The first `feature_layers - 1` convolutional layers use a stride of 2 to progressively reduce the spatial dimensions, 
            # while the final layer uses a stride of `1` to extract features without additional downsampling.
            stride = 2 if idx < feature_layers - 1 else 1
            # BatchNorm relies on "channel consistency across samples", and LayerNorm depends on "overall stability of a single sample".
            self.bns.append(tf.layers.BatchNormalization() if use_bn else None)
            self.convs.append(
                tf.layers.Conv2D(
                    filters=feature_dim,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(stride, stride),
                    padding=padding,
                    activation=tf.nn.relu,
                    name=f"layer_{layer + 1}",
                )
            )

    def __call__(self, input_tensor, training=True):
        with tf.variable_scope(None, default_name=self.name):
            tensor = input_tensor
            for conv, bn in zip(self.convs, self.bns):
                tensor = conv(tensor)
                tensor = bn(tensor, training=training) if bn is not None else tensor
            return tf.reduce_mean(tensor, axis=(-2, -3))


class PassthroughFeatureExtractor(FeatureExtractor):
    """Passthrough feature extractor.

    No feature extraction is performed, only the input is expanded, e.g.
      Input
      ├─ Flatten -> (size: [B, HWC])
      └─ wrap_feature_extractor(Input) -> (size: [B, D])
          ↓
        concat -> (size: [B, HWC + D])
    """

    def __init__(self, name: str, wrap_class=None):
        super().__init__(name=name)

        self.name = name
        if wrap_class is not None:
            self.wrap_feature_extractor = wrap_class(name=name)
        else:
            self.wrap_feature_extractor = None

    def __call__(self, input_tensor):
        output = tf.layers.Flatten()(input_tensor)
        if self.wrap_feature_extractor is not None:
            wrapped = self.wrap_feature_extractor(input_tensor)
            output = tf.concat([output, wrapped], axis=-1)
        return output


def fe_multi_layer(config: LayerwiseModelConfig, num_layers=2, use_bn=False):
    return SharedMultilayerFeatureExtractor(
        feature_layers=num_layers,
        feature_dim=config.shared_features_dim,
        name="shared_features",
        padding=config.shared_feature_extractor_padding,
        use_bn=use_bn,
    )


feature_extractors = {
    "2-layer": functools.partial(fe_multi_layer, num_layers=2),
    "3-layer": functools.partial(fe_multi_layer, num_layers=3),
    "4-layer": functools.partial(fe_multi_layer, num_layers=4),
    "2-layer-bn": functools.partial(fe_multi_layer, num_layers=2, use_bn=True),
    "3-layer-bn": functools.partial(fe_multi_layer, num_layers=3, use_bn=True),
}


def get_shared_feature_extractor(config: LayerwiseModelConfig):
    feature_extractor = config.shared_feature_extractor
    if feature_extractor in ["none", ""]:
        return None
    if feature_extractor not in feature_extractors:
        raise ValueError(f'Unknown shared feature extractor "{feature_extractor}"')
    return feature_extractors[feature_extractor](config)
