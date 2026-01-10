import torch
import torch.nn as nn
import pytest

from hypertransformer.core.feature_extractors import SimpleConvFeatureExtractor, \
    SharedMultilayerFeatureExtractor, PassthroughFeatureExtractor


# ------------------------------------------------------------
#   Class `SimpleConvFeatureExtractor` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize("feature_layers", [1, 3, 5])
def test_simple_basic(feature_layers):
    """
    [B, C, H, W] → [B, feature_dim, H_i, W_i] → [B, feature_dim, 1, 1] → [B, feature_layers*feature_dim]
    """
    B, C, H, W = 2, 3, 64, 64
    feature_dim = 8

    model = SimpleConvFeatureExtractor(
        in_channels=C,
        input_size=W,
        feature_layers=feature_layers,
        feature_dim=feature_dim,
        kernel_size=3,
        padding="valid",
        name="test_basic",
    )

    x = torch.randn(B, C, H, W)
    y = model(x)

    assert y.shape == (B, feature_layers*feature_dim)

@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("feature_layers", [1, 3, 5])
def test_simple_multi_layer_padding(padding, feature_layers):
    B, C, H, W = 2, 3, 64, 64
    feature_dim = 8

    model = SimpleConvFeatureExtractor(
        in_channels=C,
        input_size=W,
        feature_layers=feature_layers,
        feature_dim=feature_dim,
        kernel_size=3,
        padding=padding,
        name="test_multi_layer_padding",
    )

    x = torch.randn(B, C, H, W)
    y = model(x)

    assert y.shape == (B, feature_layers*feature_dim)


# ------------------------------------------------------------
#   Class `SharedMultilayerFeatureExtractor` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize("use_bn", [True, False])
@pytest.mark.parametrize("feature_layers", [1, 3, 5])
def test_shared_basic(use_bn, feature_layers):
    B, C, H, W = 4, 3, 64, 64
    feature_dim = 8

    model = SharedMultilayerFeatureExtractor(
        in_channels=C,
        feature_layers=feature_layers,
        feature_dim=feature_dim,
        kernel_size=3,
        padding="valid",
        use_bn=use_bn,
        name="test_shared_basic",
    )

    x = torch.randn(B, C, H, W)
    y = model(x)

    assert y.shape == (B, feature_dim)

@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("feature_layers", [1, 3, 5])
def test_shared_multi_layer_padding(padding, feature_layers):
    B, C, H, W = 4, 3, 64, 64
    feature_dim = 8

    model = SharedMultilayerFeatureExtractor(
        in_channels=C,
        feature_layers=feature_layers,
        feature_dim=feature_dim,
        kernel_size=3,
        padding=padding,
        name="test_shared_multi_layer_padding",
    )

    x = torch.randn(B, C, H, W)
    y = model(x)

    assert y.shape == (B, feature_dim)


# ------------------------------------------------------------
#   Class `PassthroughFeatureExtractor` Tests
# ------------------------------------------------------------


def test_pass_basic():
    B, C, H, W = 4, 3, 64, 64
    model = PassthroughFeatureExtractor(name="test_pass_basic")

    x = torch.randn(B, C, H, W)
    y = model(x)

    assert y.shape == (B, C*H*W)

class _DummyWrapFeatureExtractor(nn.Module):
    def __init__(self, name: str, out_dim: int = 7):
        super().__init__()
        self.name = name
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.ones(B, self.out_dim)

def test_pass_with_wrap_class():
    B, C, H, W = 4, 3, 64, 64
    D = 7

    model = PassthroughFeatureExtractor(
        wrap_class=lambda name: _DummyWrapFeatureExtractor(name, out_dim=D),
        name="test_pass_with_wrap_class",
    )

    x = torch.randn(B, C, H, W)
    y = model(x)

    assert y.shape == (B, C*H*W + D)
