"""Tests for `datasets.py`."""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset

from hypertransformer.core import datasets


# ------------------------------------------------------------
#   Class `TaskGenerator` Tests
# ------------------------------------------------------------


class FakeTorchDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],  # (C, H, W)
            "label": self.labels[idx],  # scalar
        }

def _make_data(batch_size: int, channels: int, image_size: int, num_labels: int) -> dict[int, np.ndarray]:
    assert batch_size % num_labels == 0
    repetitions = batch_size // num_labels

    # (batch_size, channels, image_size, image_size)
    images = np.zeros(
        (batch_size, channels, image_size, image_size),
        dtype=np.float32,
    )
    # [0,1,..,num_labels, ..., 0,1,..,num_labels]
    labels = np.array(list(range(num_labels))*repetitions, dtype=np.int32)
    ds = FakeTorchDataset(images, labels)

    return datasets.make_numpy_data(
        ds,
        batch_size=batch_size,
        num_labels=num_labels,
        samples_per_label=repetitions,
    )


@pytest.fixture(params=[
    {
        "batch_size": 4,
        "channels": 1,
        "image_size": 16,
        "num_labels": 4,
    },
    {
        "batch_size": 10,
        "channels": 3,
        "image_size": 32,
        "num_labels": 5,
    },
    {
        "batch_size": 12,
        "channels": 8,
        "image_size": 64,
        "num_labels": 6,
    },
])
def data_params(request):
    return request.param


def test_make_numpy_data(data_params):
    batch_size = data_params["batch_size"]
    channels = data_params["channels"]
    image_size = data_params["image_size"]
    num_labels = data_params["num_labels"]

    data = _make_data(batch_size, channels, image_size, num_labels)

    samples_per_label = batch_size // num_labels
    assert list(data.keys()) == list(range(num_labels))
    for label in range(num_labels):
        assert len(data[label]) == samples_per_label

def test_gen_get_batch(data_params):
    """Tests image and label generation in the `TaskGenerator`."""
    batch_size = data_params["batch_size"]
    channels = data_params["channels"]
    image_size = data_params["image_size"]
    num_labels = data_params["num_labels"]

    data = _make_data(batch_size, channels, image_size, num_labels)
    gen = datasets.TaskGenerator(data, num_labels)

    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig(
            rotation_probability=0.0,
            smooth_probability=0.0,
            contrast_probability=0.0,
        )
    )
    images, labels, classes = gen.get_batch(
        batch_size=batch_size,
        config=aug_config,
    )

    # Checks
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(classes, torch.Tensor)

    assert images.shape == (batch_size, channels, image_size, image_size)
    assert labels.shape == (batch_size,)
    assert classes.shape == (batch_size,)


# ------------------------------------------------------------
#   Class `AugmentationConfig` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        dict(
            rotation_probability=0.0,
            smooth_probability=0.0,
            contrast_probability=0.0,
            resize_probability=0.0,
            negate_probability=0.0,
            roll_probability=0.0,
        ),
        dict(
            rotation_probability=1.0,
            smooth_probability=1.0,
            contrast_probability=1.0,
            resize_probability=1.0,
            negate_probability=1.0,
            roll_probability=1.0,
        ),
    ],
)
def test_augmentation(config: dict):
    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig(**config)
    )

    images = torch.ones((4, 1, 8, 8))
    aug_config.randomize()
    out = aug_config.process(images)

    assert out.shape == images.shape
    assert out.dtype == images.dtype
