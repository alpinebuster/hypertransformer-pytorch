"""Feature extractors used to generate Layerwise models."""

import functools

from typing import Any, Optional

import tensorflow.compat.v1 as tf # pyright: ignore[reportMissingImports] # pylint:disable=import-error
import typing_extensions

from hypertransformer.core import common_ht  # pylint:disable=unused-import
from hypertransformer.core.common_ht import LayerwiseModelConfig

Protocol = typing_extensions.Protocol


class FeatureExtractor(tf.Module):
    pass


class SimpleConvFeatureExtractor(FeatureExtractor):
    """Simple convolutional feature extractor."""

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

        self.nonlinear_feature = nonlinear_feature
        self.convs = []
        def_stride = 2
        self.kernel_size = kernel_size

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
    """Simple shared convolutional feature extractor."""

    def __init__(
        self,
        feature_layers: int,
        feature_dim: int,
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
    """Passthrough feature extractor."""

    def __init__(self, name, input_size=None, wrap_class=None):
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
