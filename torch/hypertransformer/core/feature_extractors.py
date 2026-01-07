"""Feature extractors used to generate Layerwise models."""

import functools

from typing import cast, Optional

from absl import flags, logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypertransformer.core.common_ht import LayerwiseModelConfig
from hypertransformer.core.util import same_pad_2d

FLAGS = flags.FLAGS


class FeatureExtractor(nn.Module):
    """
    feature_extractor -> SimpleConvFeatureExtractor       -> Activation Feature Extractor (LogitsLayer)
    shared_features   -> SharedMultilayerFeatureExtractor -> Image Feature Extractor
                      -> PassthroughFeatureExtractor      -> FlattenLogitsLayer
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name


class SimpleConvFeatureExtractor(FeatureExtractor):
    """Simple convolutional feature extractor.

    Extract and splice multi-layer features layer by layer, e.g.:
      Input ([B, C, H, W])
      ├─ Conv1 → GAP (Global Average Pooling) → f1 ([B, feature_dim])
      ├─ Conv2 → GAP → f2 ([B, feature_dim])
      ├─ Conv3 → GAP → f3 ([B, feature_dim])
      |
      └─ concat(f1, f2, f3) ([B, [f1_1, ..., f1_C, f2_1, ..., f2_C, f3_1, ..., f3_C]])
    """

    def __init__(
        self,
        in_channels: int,
        feature_layers: int,
        feature_dim: int, # Number of Conv2D filters
        name: str,
        nonlinear_feature: bool = False,
        kernel_size: int = 3,
        input_size: Optional[int] = None,
        padding: str = "valid",
    ):
        super().__init__(name=name)

        self.feature_dim = feature_dim
        self.nonlinear_feature = nonlinear_feature
        self.kernel_size = kernel_size

        assert in_channels > 0
        assert input_size is not None
        assert feature_dim > 0

        # Stride logic (same as TF implementation)
        def_kernel_size = self.kernel_size
        def_stride = 2
        if input_size < kernel_size:
            self.kernel_size = input_size
            def_kernel_size = input_size
            def_stride = 1

        # PyTorch 1.10+ supports "same" / "valid"
        assert padding in ("same", "valid")
        self.padding = padding
        self.convs = nn.ModuleList()

        for idx in range(feature_layers):
            # The first `feature_layers - 1` convolutional layers use a stride of 2 to progressively reduce the spatial dimensions,
            # while the final layer uses a stride of `1` to extract features without additional downsampling.
            stride = def_stride if idx < feature_layers - 1 else 1
            self.kernel_size = def_kernel_size if idx < feature_layers - 1 else 1
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels if idx == 0 else feature_dim,
                    out_channels=feature_dim,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    stride=(stride, stride),
                    padding=0, # Default to `valid` mode
                    bias=True,
                )
            )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.convs:
            return None

        # BCHW
        tensor = x
        outputs = []

        for conv in self.convs:
            # TensorFlow -> BHWC, `if int(tensor.shape[1]) < kernel_size: break``
            # PyTorch    -> BCHW, spatial dims are [-2], [-1]
            if tensor.shape[-2] < self.kernel_size or tensor.shape[-1] < self.kernel_size:
                logging.info(
                    f"{self.__class__.__name__} cannot apply Conv2d at layer {len(outputs)+1}: "
                    f"spatial size {(tensor.shape[-2], tensor.shape[-1])} "
                    f"is smaller than kernel_size={self.kernel_size}. "
                    f"Check input_size, kernel_size, stride, or padding."
                )
                break

            conv = cast(nn.Conv2d, conv)
            if self.padding == "same":
                tensor = same_pad_2d(
                    tensor,
                    kernel_size=self.kernel_size,
                    stride=cast(tuple[int, int], conv.stride),
                    dilation=cast(tuple[int, int], conv.dilation),
                )
            tensor = F.conv2d(
                tensor,
                weight=conv.weight,
                bias=conv.bias,
                stride=conv.stride,
            )
            feature = tensor

            # While the output is not employing nonlinearity, layer-to-layer
            # transformations use it.
            tensor = F.relu(tensor)
            if self.nonlinear_feature:
                feature = tensor
            outputs.append(feature)

        # TF: reduce_mean(axis=(1,2)) → GAP
        #    [B, H, W, C] → [B, H_i, W_i, feature_dim] → [B, feature_dim] → [B, feature_layers*feature_dim]
        # PyTorch
        #    [B, C, H, W] → [B, feature_dim, H_i, W_i] → [B, feature_dim, 1, 1] → [B, feature_layers*feature_dim]
        outputs = [
            self.gap(feat).flatten(1)
            for feat in outputs
        ]
        return torch.cat(outputs, dim=-1)


class SharedMultilayerFeatureExtractor(FeatureExtractor):
    """Simple shared convolutional feature extractor.

    Just like a standard CNN, it only outputs the last layer of features, e.g.:
      Input ([B, C, H, W])
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
        in_channels: int = 3,
        kernel_size: int = 3,
        padding: str = "valid",
        use_bn: bool = False,
    ):
        super().__init__(name=name)

        # PyTorch 1.10+ supports "same" / "valid"
        assert padding in ("same", "valid")
        self.padding = padding
        assert feature_dim > 0

        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.use_bn = use_bn

        self.convs = nn.ModuleList()          
        self.bns = nn.ModuleList()

        for idx in range(feature_layers):
            # The first `feature_layers - 1` convolutional layers use a stride of 2 to progressively reduce the spatial dimensions, 
            # while the final layer uses a stride of `1` to extract features without additional downsampling.
            stride = 2 if idx < feature_layers - 1 else 1
            # BatchNorm relies on "channel consistency across samples", and LayerNorm depends on "overall stability of a single sample".
            if use_bn:
                self.bns.append(nn.BatchNorm2d(num_features=feature_dim))
            else:
                self.bns.append(nn.Identity())

            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels if idx == 0 else feature_dim,
                    out_channels=feature_dim,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    stride=(stride, stride),
                    padding=0, # Default to `valid` mode
                    bias=True,
                )
            )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        tensor = x
        for idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if tensor.shape[-2] < self.kernel_size or tensor.shape[-1] < self.kernel_size:
                logging.info(
                    f"{self.__class__.__name__} cannot apply Conv2d at layer {idx+1}: "
                    f"spatial size {(tensor.shape[-2], tensor.shape[-1])} "
                    f"is smaller than kernel_size={self.kernel_size}. "
                    f"Check input_size, kernel_size, stride, or padding."
                )
                break

            conv = cast(nn.Conv2d, conv)
            if self.padding == "same":
                tensor = same_pad_2d(
                    tensor,
                    kernel_size=self.kernel_size,
                    stride=cast(tuple[int, int], conv.stride),
                    dilation=cast(tuple[int, int], conv.dilation),
                )
            tensor = F.conv2d(
                tensor,
                weight=conv.weight,
                bias=conv.bias,
                stride=conv.stride,
            )
            tensor = bn(tensor)
        return self.gap(tensor).flatten(1) 


class PassthroughFeatureExtractor(FeatureExtractor):
    """Passthrough feature extractor.

    No feature extraction is performed, only the input is expanded, e.g.
      Input
      |  x -> [B, C, H, W]
      ├─ Flatten -> [B, CHW]
      ├───├─ wrap_feature_extractor(Input) -> [B, D]
      |   |     ↓
      |   ↓  concat -> [B, CHW + D]
      └─ [B, CHW]
    """

    def __init__(self, name: str, wrap_class=None):
        super().__init__(name=name)

        self.name = name
        self.wrap_feature_extractor: Optional[nn.Module] = None
        if wrap_class is not None:
            # `wrap_class` must be a subclass of `nn.Module`
            self.wrap_feature_extractor = wrap_class(name=name)
        else:
            self.wrap_feature_extractor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C*H*W]
        output = torch.flatten(x, start_dim=1)
        if self.wrap_feature_extractor is not None:
            wrapped = self.wrap_feature_extractor(x)
            # [B, CHW + D]
            output = torch.cat([output, wrapped], dim=-1)
        return output


class SharedHead(nn.Module):
    def __init__(
        self,
        shared_features_dim: int,
        real_class_min: int,
        real_class_max: int,
        label_smoothing: float = 0.,
    ):
        super().__init__()
        self.real_class_min = real_class_min
        self.real_class_max = real_class_max
        self.total_classes = real_class_max - real_class_min + 1
        self.label_smoothing = label_smoothing

        # Default to `FLAGS.shared_features_dim`
        self.fc = nn.Linear(shared_features_dim, self.total_classes)

    def forward(
        self,
        shared_features: Optional[torch.Tensor],   # (B, shared_features_dim)
        real_classes: torch.Tensor,      # (B,)
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if shared_features is not None:
            logits = self.fc(shared_features)  # (B, total_classes)
            # Class index normalization
            classes = real_classes - self.real_class_min  # (B,)
            classes = classes.long()

            # loss (TF softmax_cross_entropy ≈ PyTorch cross_entropy)
            loss = F.cross_entropy(
                logits,
                classes,
                label_smoothing=self.label_smoothing,
            )

            # accuracy
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == classes).float().mean()

            return loss, accuracy
        else:
            return None, None


def fe_multi_layer(config: LayerwiseModelConfig, num_layers=2, use_bn=False):
    return SharedMultilayerFeatureExtractor(
        feature_layers=num_layers,
        feature_dim=config.shared_features_dim,
        in_channels=config.shared_input_dim,
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
