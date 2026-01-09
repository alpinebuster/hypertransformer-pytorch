"""Training library and binary."""

import dataclasses
import functools
import os
import random
import glob

from typing import Optional

from absl import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from hypertransformer.core import common_ht
from hypertransformer.core.common_ht import LayerwiseModelConfig, DatasetConfig
from hypertransformer.core import datasets


@dataclasses.dataclass
class ModelState:
    """Model state."""

    loss: Optional[torch.Tensor] = None


def _make_augmentation_config(data_config: DatasetConfig, num_labels: int):
    """Returns dataset augmentation configuration."""
    random_config = datasets.RandomizedAugmentationConfig(
        rotation_probability=data_config.rotation_probability,
        smooth_probability=data_config.smooth_probability,
        contrast_probability=data_config.contrast_probability,
        resize_probability=data_config.resize_probability,
        negate_probability=data_config.negate_probability,
        roll_probability=data_config.roll_probability,
        angle_range=data_config.angle_range,
        rotate_by_90=data_config.rotate_by_90,
    )
    if data_config.per_label_augmentation:
        return datasets.AugmentationConfig(
            children=[
                datasets.AugmentationConfig(random_config=random_config)
                for _ in range(num_labels)
            ]
        )
    else:
        return datasets.AugmentationConfig(random_config=random_config)


def _convert_bool(arr: np.ndarray):
    return arr.astype(np.int8) * 255

def _load_cache(data_config: DatasetConfig) -> Optional[dict[int, np.ndarray]]:
    """Loads cached dataset from a saved NumPy array."""
    folder = os.path.join(data_config.cache_path, data_config.dataset_name)

    pattern = os.path.join(data_config.cache_path, f"*{data_config.dataset_name}_ch_first*.npy")
    files = glob.glob(pattern)
    path = files[0] if files else ""

    logging.info(
        f"[DDP] global_rank={dist.get_rank()} >>> "
        f'Looking for cache in "{data_config.cache_path}..."'
    )
    if os.path.exists(path):
        # Reading a NumPy cache.
        with open(path, "rb") as dev:
            data = np.load(dev)

        if len(data.shape) < 4:
            data = np.expand_dims(data, axis=-3)
        # Converting a 5D tensor [all_classes, all_samples, C, H, W] to a dictionary by label.
        if data.dtype == bool:
            return {k: _convert_bool(data[k]) for k in range(data.shape[0])}
        else:
            return {k: data[k] for k in range(data.shape[0])}
    elif os.path.exists(folder):
        # Reading from a folder with NumPy cache files.
        names = os.listdir(folder)
        output = {}
        index = 0
        for name in sorted(names):
            # Each file contains a list of image sets for different labels.
            # File names are sorted to keep a proper label order.
            with open(os.path.join(folder, name), "rb") as data_file:
                file_records = np.load(data_file, allow_pickle=True)
            for record in file_records:
                output[index] = record
                index += 1
        return output
    logging.info(
        f"[DDP] global_rank={dist.get_rank()} >>> "
        f"No cache files for {data_config.dataset_name} found. Falling back "
        "to torch dataset."
    )
    return None

def make_numpy_array(
    data_config: DatasetConfig,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Makes a NumPy array for given dataset configuration.
    
    Returns:
       Dictionary mapping labels to tensors containing all samples (samples_per_label, C, H, W).

       Dict[int, np.ndarray]:
         int -> all_classes
         np.ndarray -> [all_samples, C, H, W]
    """
    output = None

    ds = data_config.ds
    if ds is None:
        output = _load_cache(data_config)
        if output is None:
            raise ValueError(f"Dataset {data_config.dataset_name} not found and no cache available.")

    dataset_info = data_config.dataset_info
    if dataset_info is None:
        dataset_info = datasets.get_dataset_info(data_config.dataset_name)

    if output is None and ds is not None:
        assert dataset_info.num_samples_per_label is not None
        output = datasets.make_numpy_data(
            ds=ds,
            batch_size=batch_size,
            num_labels=dataset_info.num_labels,
            samples_per_label=dataset_info.num_samples_per_label,
            transpose=dataset_info.transpose_images,
        )
    assert output is not None
    if data_config.shuffle_labels_seed > 0:
        keys = list(output.keys())
        orig_keys = keys[:]
        random.seed(data_config.shuffle_labels_seed)
        random.shuffle(keys)
        output = {orig_keys[i]: output[keys[i]] for i in range(len(keys))}

    return output


def _resize(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if imgs.dim() == 2: # HW
        imgs = imgs.unsqueeze(0).unsqueeze(0)
    elif imgs.dim() == 3:
        if imgs.shape[-1] <= 4: # HWC (TF Style)
            imgs = imgs.permute(2, 0, 1).unsqueeze(0)
        else: # CHW
            imgs = imgs.unsqueeze(0)
    elif imgs.dim() == 4:
        if imgs.shape[-1] <= 4:  # BHWC
            imgs = imgs.permute(0, 3, 1, 2)

    return F.interpolate(
        imgs,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def _make_dataset_helper_unbalanced(
    numpy_arr: dict[int, np.ndarray],
    batch_size: int,
    image_size: int,
    num_labels: int,
    data_config: DatasetConfig,
    always_same_labels=False,
) -> datasets.SupervisedSample:
    """Helper function for creating a dataset.

       `numpy_arr = make_numpy_array(data_config, batch_size)`
    """
    config = _make_augmentation_config(data_config=data_config, num_labels=num_labels)

    gen = datasets.TaskGenerator(
        numpy_arr,
        num_labels=num_labels,
        use_label_subset=data_config.use_label_subset,
        always_same_labels=always_same_labels,
    )
    config.randomize()
    images, labels, classes = gen.get_batch(
        batch_size=batch_size,
        config=config,
        num_unlabeled_per_class=data_config.num_unlabeled_per_class,
    )

    if image_size is not None:
        images = _resize(images, image_size)
    images = images / 128.0 - 1.0

    sample: datasets.SupervisedSample = (images, labels, classes)
    return sample


def _make_dataset_helper_balanced(
    numpy_arr: dict[int, np.ndarray],
    batch_sizes: list[int],
    num_unlabeled_per_class: list[int],
    image_size,
    num_labels: int,
    data_config: DatasetConfig,
    always_same_labels=False,
) -> list[datasets.SupervisedSample]:
    """Helper function for creating a balanced dataset.

       `numpy_arr = make_numpy_array(data_config, batch_sizes[0])`
    """
    config = _make_augmentation_config(data_config=data_config, num_labels=num_labels)

    gen = datasets.TaskGenerator(
        numpy_arr,
        num_labels=num_labels,
        use_label_subset=data_config.use_label_subset,
        always_same_labels=always_same_labels,
    )
    config.randomize()
    images_labels = gen.get_batches(
        batch_sizes=batch_sizes,
        config=config,
        num_unlabeled_per_class=num_unlabeled_per_class,
    )
    output: list[datasets.SupervisedSample] = []
    for images, labels, classes in images_labels:
        if image_size is not None:
            images = _resize(images, image_size)
        # [0, 255] → [0, 1.992] → [-1.0, 0.992]
        images = images / 128.0 - 1.0
        output.append((images, labels, classes))

    return output


def _get_class_bounds(data_config: DatasetConfig):
    if data_config.use_label_subset is None or callable(data_config.use_label_subset):
        return None, None
    return min(data_config.use_label_subset), max(data_config.use_label_subset)


def _make_dataset_unbalanced(
    numpy_arr: dict[int, np.ndarray],
    model_config: LayerwiseModelConfig,
    data_config: DatasetConfig,
    shuffle_labels=True,
):
    """Creates data for Transformer and CNN.

    Arguments:
      model_config: Model configuration.
      data_config: Dataset configuration.
      shuffle_labels: True if should subsample random labels from
          `data_config.use_label_subset` for each new mini-dataset.

    Returns:
      `DatasetSamples` structure containing Transformer and CNN samples.
    """
    batch_size = model_config.num_transformer_samples
    batch_size += model_config.num_cnn_samples

    assert model_config.image_size
    sample = _make_dataset_helper_unbalanced(
        numpy_arr=numpy_arr,
        batch_size=batch_size,
        image_size=model_config.image_size,
        num_labels=model_config.num_labels,
        data_config=data_config,
        always_same_labels=not shuffle_labels,
    )
    images, labels, classes = sample

    transformer_samples = model_config.num_transformer_samples
    transformer_images = images[:transformer_samples]
    if len(transformer_images.shape) == 3:
        transformer_images = transformer_images.unsqueeze(dim=-3)
    cnn_images = images[transformer_samples:]
    if len(cnn_images.shape) == 3:
        cnn_images = cnn_images.unsqueeze(dim=-3)

    real_class_min, real_class_max = _get_class_bounds(data_config)

    return common_ht.DatasetSamples(
        transformer_images=transformer_images,
        transformer_labels=labels[:transformer_samples],
        transformer_real_classes=classes[:transformer_samples],
        cnn_images=cnn_images,
        cnn_labels=labels[transformer_samples:],
        cnn_real_classes=classes[transformer_samples:],
        real_class_min=real_class_min,
        real_class_max=real_class_max,
    )


def _make_dataset_balanced(
    numpy_arr: dict[int, np.ndarray],
    model_config: LayerwiseModelConfig,
    data_config: DatasetConfig,
    shuffle_labels: bool = True,
):
    """Creates data for Transformer and CNN.

    Arguments:
      model_config: Model configuration.
      data_config: Dataset configuration.
      shuffle_labels: True if should subsample random labels from
          `data_config.use_label_subset` for each new mini-dataset.

    Returns:
      `DatasetSamples` structure containing Transformer and CNN samples.
    """
    batch_sizes = [model_config.num_transformer_samples, model_config.num_cnn_samples]
    # Removing labels only from the Transformer batch.
    num_unlabeled_per_class = [data_config.num_unlabeled_per_class, 0]

    batches = _make_dataset_helper_balanced(
        numpy_arr=numpy_arr,
        batch_sizes=batch_sizes,
        num_unlabeled_per_class=num_unlabeled_per_class,
        image_size=model_config.image_size,
        num_labels=model_config.num_labels,
        data_config=data_config,
        always_same_labels=not shuffle_labels,
    )

    transformer_images, transformer_labels, transformer_classes = batches[0]
    cnn_images, cnn_labels, cnn_classes = batches[1]

    # BCHW
    if len(transformer_images.shape) == 3:
        transformer_images = transformer_images.unsqueeze(dim=-3)
    if len(cnn_images.shape) == 3:
        cnn_images = cnn_images.unsqueeze(dim=-3)

    real_class_min, real_class_max = _get_class_bounds(data_config)

    return common_ht.DatasetSamples(
        transformer_images=transformer_images,
        transformer_labels=transformer_labels,
        transformer_real_classes=transformer_classes,
        cnn_images=cnn_images,
        cnn_labels=cnn_labels,
        cnn_real_classes=cnn_classes,
        real_class_min=real_class_min,
        real_class_max=real_class_max,
    )


def make_dataset(
    numpy_arr: dict[int, np.ndarray],
    model_config: LayerwiseModelConfig,
    data_config: DatasetConfig,
    **kwargs,
):
    """Makes dataset given dataset and model configuration.
    
    1) Balanced:
       The number of samples for each category in each batch is the same (or as much as possible the same), and within the batch, they are clearly organized by "category".

       e.g.
          Transformer: [0,1,2,3,4, ...]
          CNN: [0,1,2,3,4, ...]

    2) Unbalanced:
       In a batch, only the total number of samples is guaranteed, and the occurrence frequency of each category is not guaranteed to be consistent.

       e.g.
          Transformer: [0, 0, 0, 1, 1, 3]
          CNN: [2, 2, 2, 4, 4]
    """
    augment = functools.partial(
        datasets.augment_images,
        augment_individually=data_config.augment_individually,
    )

    if data_config.balanced_batches:
        # The two batches are "generated separately".
        dataset_maker = _make_dataset_balanced
    else:
        dataset_maker = _make_dataset_unbalanced

    output = dataset_maker(
        numpy_arr=numpy_arr,
        model_config=model_config,
        data_config=data_config,
        **kwargs,
    )
    if data_config.apply_image_augmentations:
        assert model_config.image_size is not None
        image_size = model_config.image_size
        output = dataclasses.replace(
            output,
            transformer_images=augment(output.transformer_images, image_size),
            cnn_images=augment(output.cnn_images, image_size),
        )

    return output
