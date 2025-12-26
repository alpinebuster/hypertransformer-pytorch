"""
Call from the `torch` root dir:
    `pytest -s ./hypertransformer/core/test_feature_extractors.py`
"""

import torch

from hypertransformer.core.feature_extractors import SimpleConvFeatureExtractor

print("\nTORCH VERSION IN TEST:", torch.__version__)


def test_basic():
    """
    [B, C, H, W] → [B, feature_dim, H_i, W_i] → [B, feature_dim, 1, 1] → [B, feature_layers*feature_dim]
    """
    x = torch.randn(2, 3, 32, 32)

    model = SimpleConvFeatureExtractor(
        in_channels=x.shape[1],
        input_size=x.shape[-1],
        feature_layers=1,
        feature_dim=8,
        kernel_size=3,
        padding="same",
        name="test_basic",
    )

    y = model(x)

    assert y.shape == (2, 8)


def test_multi_layer_valid():
    x = torch.randn(2, 3, 32, 32)

    model = SimpleConvFeatureExtractor(
        in_channels=x.shape[1],
        input_size=x.shape[-1],
        feature_layers=3,
        feature_dim=8,
        kernel_size=3,
        padding="valid",
        name="test_multi_layer_valid",
    )

    y = model(x)

    assert y.shape == (2, 24)


def test_multi_layer_same():
    x = torch.randn(2, 3, 32, 32)

    model = SimpleConvFeatureExtractor(
        in_channels=x.shape[1],
        input_size=x.shape[-1],
        feature_layers=3,
        feature_dim=8,
        kernel_size=3,
        padding="same",
        name="test_multi_layer_same",
    )

    y = model(x)

    assert y.shape == (2, 24)
