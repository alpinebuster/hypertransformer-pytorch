"""Tests for `util.py`."""

from typing import Union
from collections import namedtuple
import pytest
import torch

from hypertransformer.core import util
from hypertransformer.core.util import _TransformerIO, SimpleTransformerIO, \
    SeparateTransformerIO, JointTransformerIO

IOParams = namedtuple("IOParams", [
    "batch_size",
    "embedding_dim",
    "num_weights",
    "num_labels",
    "image_size",
])


@pytest.fixture(params=[
    (1, 4, 16, 4, 32),
    (8, 8, 128, 6, 64),
    (16, 16, 256, 8, 128),
])
def transformer_io_params(request) -> IOParams:
    """
    Fixture for `*TransformerIO` test parameters.

    Returns a dict with keys:
        batch_size, embedding_dim, num_weights, num_labels, image_size
    """
    return IOParams(*request.param)

class TestUtil:
    """Tests for util.py (PyTorch version)."""

    def _encode_samples(
        self,
        batch_size: int,
        image_size: int,
        io: Union[SimpleTransformerIO, SeparateTransformerIO, JointTransformerIO],
    ) -> torch.Tensor:
        """Creates empty images and labels and encodes for Transformer."""
        # [batch_size, image_size, image_size, channels]
        images: torch.Tensor = torch.zeros(
            batch_size, image_size, image_size, 1, dtype=torch.float32
        )
        labels: torch.Tensor = torch.zeros(batch_size, dtype=torch.long)
        return io.encode_samples(images, labels)

    def _check_weights(
        self,
        encoded: torch.Tensor,
        num_weights: int,
        image_size: int,
        io: Union[SimpleTransformerIO, SeparateTransformerIO, JointTransformerIO],
    ) -> None:
        """Decodes weights and checks their number and their shapes."""
        weights: list[torch.Tensor] = io.decode_weights(encoded)

        assert isinstance(weights, list)
        assert len(weights) == num_weights

        for weight in weights:
            assert weight.shape == (image_size**2,)

    def test_simple_transformer_io(
        self,
        transformer_io_params: IOParams,
    ) -> None:
        """Tests SimpleTransformerIO Transformer adapter."""
        batch_size = transformer_io_params.batch_size
        embedding_dim = transformer_io_params.embedding_dim
        num_weights = transformer_io_params.num_weights
        num_labels = transformer_io_params.num_labels
        image_size = transformer_io_params.image_size

        io = SimpleTransformerIO(
            num_labels=num_labels,
            num_weights=num_weights,
            embedding_dim=embedding_dim,
            weight_block_size=image_size**2,
        )
        encoded: torch.Tensor = self._encode_samples(
            batch_size, image_size, io
        )

        assert encoded.shape == (
            batch_size,
            embedding_dim + (image_size**2),
        )
        self._check_weights(
            encoded,
            num_weights=num_weights,
            image_size=image_size,
            io=io,
        )

    def test_joint_transformer_io(
        self,
        transformer_io_params: IOParams,
    ) -> None:
        """Tests JointTransformerIO Transformer adapter."""
        batch_size = transformer_io_params.batch_size
        embedding_dim = transformer_io_params.embedding_dim
        num_weights = transformer_io_params.num_weights
        num_labels = transformer_io_params.num_labels
        image_size = transformer_io_params.image_size

        io = JointTransformerIO(
            num_labels=num_labels,
            num_weights=num_weights,
            embedding_dim=embedding_dim,
            weight_block_size=image_size**2,
        )
        encoded: torch.Tensor = self._encode_samples(
            batch_size, image_size, io
        )

        expected_shape: tuple[int, int] = (
            batch_size + num_weights,
            embedding_dim + (image_size**2),
        )
        assert encoded.shape == expected_shape
        self._check_weights(
            encoded,
            num_weights=num_weights,
            image_size=image_size,
            io=io,
        )
