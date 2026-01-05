"""Tests for `train_lib.py`."""

import pytest
import numpy as np

from hypertransformer.core import common_ht
from hypertransformer.core import train_lib
from hypertransformer.core import datasets


@pytest.mark.parametrize(
    "num_transformer_samples,num_cnn_samples,num_samples_per_label,num_labels,channels,image_size",
    [
        (8, 8, 8, 4, 1, 16), # num_transformer_samples < num_samples_per_label and num_cnn_samples < num_samples_per_label
        (2, 6, 10, 5, 3, 32),
        (96, 16, 128, 6, 3, 64),
    ],
)
def test_make_dataset_helper(
    num_transformer_samples: int,
    num_cnn_samples: int,
    num_samples_per_label: int,
    num_labels: int,
    channels: int,
    image_size: int,
):
    """Tests `make_dataset_helper` function."""
    batch_size = num_samples_per_label * num_labels
    assert batch_size % num_labels == 0
    repetitions = batch_size // num_labels

    # (batch_size, channels, image_size, image_size)
    images = np.zeros(
        (batch_size, channels, image_size, image_size),
        dtype=np.float32,
    )
    # [0,1,..,num_labels, ..., 0,1,..,num_labels]
    labels = np.array(list(range(num_labels))*repetitions, dtype=np.int32)
    ds = datasets.HTDataset(images, labels)

    model_config = common_ht.LayerwiseModelConfig(
        num_transformer_samples=num_transformer_samples,
        num_cnn_samples=num_cnn_samples,
        image_size=image_size,
    )
    dataset_info = common_ht.DatasetInfo(
        num_labels=num_labels,
        num_samples_per_label=num_samples_per_label,
        transpose_images=False,
    )
    data_config = common_ht.DatasetConfig(
        dataset_name="dataset",
        ds=ds,
        dataset_info=dataset_info,
    )

    numpy_arr = train_lib.make_numpy_array(
        data_config,
        batch_size=model_config.num_transformer_samples,
    )
    samples = train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=model_config,
        data_config=data_config,
        shuffle_labels=True,
    )
    outputs = {
        "train_images": samples.transformer_images,
        "train_labels": samples.transformer_labels,
        "cnn_images": samples.cnn_images,
        "cnn_labels": samples.cnn_labels,
    }

    assert outputs["train_images"].shape == (num_transformer_samples, channels, image_size, image_size)
    assert outputs["train_labels"].shape == (num_transformer_samples,)
    assert outputs["cnn_images"].shape == (num_cnn_samples, channels, image_size, image_size)
    assert outputs["cnn_labels"].shape == (num_cnn_samples,)
