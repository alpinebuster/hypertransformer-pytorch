"""Task generator using basic datasets."""

import dataclasses
import functools
import random
import math

from typing import Any, Callable, Generator, \
    Optional, Tuple, Union, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from hypertransformer.core import common_ht

DatasetInfo = common_ht.DatasetInfo

UseLabelSubset = Union[list[int], Callable[[], list[int]]]
LabelGenerator = Generator[Tuple[int|np.ndarray, int], None, None]
SupervisedSamplesNumpy = Tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]
SupervisedSample = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


# Range of the random alpha value controlling contrast enhancement augmentation.
ALPHA_MIN = 0.5
ALPHA_MAX = 2.5
# Maximum resize fraction in the resize augmentation (0.3 means 70%-100% of the
# original size).
MAX_RESIZE = 0.3

# Default image augmentation parameters
HUE_MAX_DELTA = 0.2
BRIGHTNESS_MAX_DELTA = 0.3
LOWER_CONTRAST = 0.8
UPPER_CONTRAST = 1.2
MIN_CROP_FRACTION = 0.6


def _get_random_bounding_box(
    hw: torch.Tensor,
    aspect_ratio: float = 1.0,
    min_crop_fraction: float = 0.5,
    new_height: Optional[Union[float, torch.Tensor]] = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Random bounding box with aspect_ratio and within (0, 0, hw[:,0], hw[:,1]).

    Args:
        hw: Tensor of shape (..., 2), last dimension is (height, width)
        aspect_ratio: Desired width/height ratio
        min_crop_fraction: Minimum fraction of original size
        new_height: Optional fixed new height
        dtype: Output dtype (torch.int32, torch.int64, torch.float32, etc.)

    Returns:
        Tensor: Bounding boxes of shape (..., 4), format (y1, x1, y2, x2)
    """
    hw = hw.float()
    h = hw[..., 0] # height → y
    w = hw[..., 1] # width  → x
    if new_height is None:
        min_height = torch.min(
            h * min_crop_fraction,
            w * min_crop_fraction/aspect_ratio,
        )
        max_height = torch.min(h, w/aspect_ratio)
        new_h = min_height + (max_height - min_height) * torch.rand_like(h)
    else:
        if isinstance(new_height, torch.Tensor):
            new_h = new_height.float()
        else:
            new_h = torch.tensor(new_height, dtype=torch.float32)

    new_hw = torch.stack([new_h, new_h/aspect_ratio], dim=-1)
    if dtype in [torch.int32, torch.int64]:
        new_hw = torch.round(new_hw)

    minval = torch.zeros_like(hw)
    maxval = hw - new_hw

    # Random top-left coordinate
    tl = minval + (maxval - minval) * torch.rand_like(hw)
    if dtype in [torch.int32, torch.int64]:
        tl = torch.round(tl)

    br = tl + new_hw
    boxes = torch.concat((tl, br), dim=-1)
    if dtype in [torch.int32, torch.int64]:
        boxes = torch.round(boxes)

    return boxes.to(dtype)

def _crop_and_resize(
    images: torch.Tensor, # (C, H, W) or (N, C, H, W)
    bboxes: torch.Tensor, # (N, 4) -> Normalized (y1, x1, y2, x2)
    target_size: Union[int, Tuple[int, int], list[int]],
    methods: Union[str, Callable] = "BILINEAR",
) -> torch.Tensor:
    """Does crop and resize given normalized boxes.

    Args:
       images: (B, C, H, W) or (C, H, W)
       bboxes: (B, 4) normalized [y1, x1, y2, x2]
       target_size: int or (H, W)
       methods: "bilinear" | "nearest" | (method1, method2)

    Returns:
       (B, C, target_H, target_W) or (C, target_H, target_W)
    """
    bboxes = bboxes.float()
    if not isinstance(target_size, (tuple, list)):
        target_size = [target_size, target_size]

    squeeze = False
    if images.dim() == 3:
        # (C, H, W) → (1, C, H, W)
        images = images.unsqueeze(dim=0)
        # bbox (4,) → (1, 4)
        bboxes = bboxes.unsqueeze(dim=0)
        squeeze = True
    B, _, H, W = images.shape

    # --- normalized → absolute ---
    # roi_align format: (batch_idx, x1, y1, x2, y2) ∈ pixel coords
    # [y1, x1, y2, x2] ∈ [0, 1] → (x, y) ∈ [0, W-1] × [0, H-1]
    y1, x1, y2, x2 = bboxes.unbind(dim=-1)
    x1 = x1 * (W-1)
    x2 = x2 * (W-1)
    y1 = y1 * (H-1)
    y2 = y2 * (H-1)

    # roi_align format: (batch_idx, x1, y1, x2, y2)
    batch_idx = torch.arange(B, device=images.device, dtype=torch.float32)
    rois = torch.stack([batch_idx, x1, y1, x2, y2], dim=1)

    # (num_rois, C, target_H, target_W)
    out = roi_align( # type:ignore
        input=images,
        boxes=rois,
        output_size=target_size,
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True,
    )
    if methods == "nearest":
        # roi_align only supports "bilinear"
        out = torch.nn.functional.interpolate(
            out, size=target_size, mode="nearest"
        )

    # When `images.dim() == 3`
    if squeeze:
        # (1, C, H, W) → (C, H, W)
        out = out.squeeze(0)
    return out

def _random_crop_and_resize(
    images: torch.Tensor,
    target_size: Union[int, Tuple[int, int], list[int]],
    min_crop_fraction: float = 0.5,
    crop_size: Optional[float] = None,
    methods: Union[str, Callable] = "BILINEAR",
) -> torch.Tensor:
    """All tensors are cropped to the same size and resized to target size."""
    if not isinstance(target_size, (tuple, list)):
        target_size = [target_size, target_size]

    aspect_ratio = target_size[0] / target_size[1]

    # Batch size from images
    if images.dim() == 3:
        batch_size = 1
    else:
        batch_size = images.shape[0]

    bboxes = _get_random_bounding_box(
        hw=torch.ones((batch_size, 2), device=images.device),
        aspect_ratio=aspect_ratio,
        min_crop_fraction=min_crop_fraction,
        new_height=crop_size,
        dtype=torch.float32,
    )
    return _crop_and_resize(images, bboxes, target_size, methods)

def augment_images(
    images: torch.Tensor,
    image_size: int | Tuple[int, int],
    augment_individually: bool = False,
    hue_delta: Optional[float] = None,
    brightness_delta: Optional[float] = None,
    contrast: Optional[Tuple[float, float]] = None,
    min_crop_fraction: Optional[float] = None,
) -> torch.Tensor:
    """Standard image augmentation. Input / output range: [-1, 1]"""
    assert image_size is not None

    if hue_delta is None:
        hue_delta = HUE_MAX_DELTA
    if brightness_delta is None:
        brightness_delta = BRIGHTNESS_MAX_DELTA
    if contrast is None:
        contrast = (LOWER_CONTRAST, UPPER_CONTRAST)
    if min_crop_fraction is None:
        min_crop_fraction = MIN_CROP_FRACTION

    # === [-1,1] → uint8 ===
    images = ((images + 1.0) * 128.0).clamp(0, 255).to(torch.uint8)

    def _augment(image_tensor: torch.Tensor) -> torch.Tensor:
        # image_tensor: (C, H, W) or (B, C, H, W)
        out = TF.hflip(image_tensor) if torch.rand(()) < 0.5 else image_tensor
        # hue_factor=`2*torch.rand(())-1` ∈ [-1, 1) -> [-hue_delta, +hue_delta]
        out = TF.adjust_hue(out, hue_factor=hue_delta * (2*torch.rand(size=()).item() - 1))
        # brightness_factor ∈ [1 - brightness_delta, 1 + brightness_delta]
        out = TF.adjust_brightness(
            out, brightness_factor=1.0 + brightness_delta * (2*torch.rand(size=()).item() - 1)
        )
        # contrast_factor ∈ [contrast[0], contrast[1]]
        out = TF.adjust_contrast(
            out,
            contrast_factor=contrast[0] + (contrast[1] - contrast[0])*torch.rand(size=()).item(),
        )
        return out

    if augment_individually:
        # images: (N, C, H, W)
        images = torch.stack([_augment(img) for img in images], dim=0)
    else:
        images = _augment(images)

    images = images.float()/128.0 - 1.0
    return _random_crop_and_resize(
        images, image_size, min_crop_fraction=min_crop_fraction
    )


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """Returns basic information about the dataset."""
    if dataset_name == "emnist":
        return DatasetInfo(
            num_labels=62, num_samples_per_label=1500, transpose_images=True
        )
    elif dataset_name == "fashion_mnist":
        return DatasetInfo(
            num_labels=10, num_samples_per_label=3000, transpose_images=False
        )
    elif dataset_name == "kmnist":
        return DatasetInfo(
            num_labels=10, num_samples_per_label=3000, transpose_images=False
        )
    elif dataset_name == "omniglot":
        return DatasetInfo(
            num_labels=1623, num_samples_per_label=20, transpose_images=False
        )
    elif dataset_name == "miniimagenet":
        return DatasetInfo(
            num_labels=100, num_samples_per_label=600, transpose_images=False
        )
    elif dataset_name == "tieredimagenet":
        return DatasetInfo(
            num_labels=608,
            # This dataset has a variable number of samples per class
            num_samples_per_label=None,
            transpose_images=False,
        )
    else:
        raise ValueError(f'Dataset "{dataset_name}" is not supported.')


@dataclasses.dataclass
class RandomizedAugmentationConfig:
    """Specification of random augmentations applied to the images."""

    rotation_probability: float = 0.5
    smooth_probability: float = 0.3
    contrast_probability: float = 0.3
    resize_probability: float = 0.0
    negate_probability: float = 0.0
    roll_probability: float = 0.0
    angle_range: float = 180.0
    roll_range: float = 0.3
    rotate_by_90: bool = False

@dataclasses.dataclass
class _RandomValue:
    value: Optional[float] = None

    def assign_bool(self, prob: float):
        self.value = random.random() < prob

    def assign_uniform(self, scale: float):
        self.value = random.uniform(-scale, scale)

class _RandomRoll(nn.Module):
    def __init__(self, prob=0.0, roll_x=0., roll_y=0.):
        super().__init__()
        self.prob = prob
        self.roll_x = roll_x
        self.roll_y = roll_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(size=[1]).item() > self.prob:
            return x

        if x.dim() == 4:
            _, _, H, W = x.shape
            dims = (2, 3)
        elif x.dim() == 3:
            _, H, W = x.shape
            dims = (1, 2)
        else:
            raise ValueError("Input tensor must be 3D or 4D")

        dx = int(self.roll_x * H)
        dy = int(self.roll_y * W)
        return torch.roll(x, shifts=(dx, dy), dims=dims)

class _RandomResizedCropSameSize(nn.Module):
    def __init__(self, prob=0.0, size=1.0):
        super().__init__()
        self.prob = prob
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.prob:
            return x

        # BCHW, H = W
        H = int(x.shape[-1])
        # re_size ∈ [H*(1-MAX_RESIZE), H]
        re_size = H * (1 - MAX_RESIZE / 2 + MAX_RESIZE * self.size / 2)

        return T.RandomResizedCrop(
            size=(H, H),
            scale=(re_size, re_size),
        )(x)

@dataclasses.dataclass
class AugmentationConfig:
    """Configuration of the image augmentation generator.

           TF1:                    PyTorch:
       +----------------+      +----------------+
       | randomize_op() | ---> | randomize()    |  ← Step-level randomization
       +----------------+      +----------------+
               |                       |
               v                       v
       +----------------+      +----------------+
       | process(images)| ---> | process(images)|  ← Read the random state
       +----------------+      +----------------+
               |                       |
          output images          output images
    """

    random_config: Optional[RandomizedAugmentationConfig] = None

    rotate: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    smooth: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    contrast: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    negate: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    roll: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    resize: _RandomValue = dataclasses.field(default_factory=_RandomValue)

    angle: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    alpha: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    size: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    roll_x: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    roll_y: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    rotate_90_times: _RandomValue = dataclasses.field(default_factory=_RandomValue)

    children: list["AugmentationConfig"] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.children:
            return

        for name in ["rotate", "smooth", "contrast", "negate", "resize", "roll"]:
            attr = getattr(self, name)
            if attr.value is None:
                attr.value = False # Default bool: False
        for name in [
            "angle",
            "alpha",
            "size",
            "roll_x",
            "roll_y",
            "rotate_90_times",
        ]:
            attr = getattr(self, name)
            if attr.value is None:
                attr.value = 0.0  # Default float: 0.0

    def randomize(self):
        if self.children:
            for child in self.children:
                child.randomize()
            return

        if self.random_config is None:
            return

        cfg = self.random_config

        self.rotate.assign_bool(cfg.rotation_probability)
        self.smooth.assign_bool(cfg.smooth_probability)
        self.contrast.assign_bool(cfg.contrast_probability)
        self.negate.assign_bool(cfg.negate_probability)
        self.resize.assign_bool(cfg.resize_probability)
        self.roll.assign_bool(cfg.roll_probability)

        self.angle.assign_uniform(cfg.angle_range)

        self.size.assign_uniform(1.0)
        self.alpha.assign_uniform(1.0)

        # [-roll_range, +roll_range)
        self.roll_x.assign_uniform(cfg.roll_range)
        self.roll_y.assign_uniform(cfg.roll_range)

        # TF: scale=2.0 → U(-2, 2)
        self.rotate_90_times.assign_uniform(2.0)

    def _build_transform(self) -> T.Compose:
        cfg = self.random_config
        transforms: list[Callable[[torch.Tensor], torch.Tensor]] = []
        if cfg is None:
            return T.Compose(transforms)

        if cfg.rotate_by_90 and self.rotate_90_times.value:
            # -> 0..3
            k = int(math.floor(self.rotate_90_times.value + 2.0)) % 4
            angle = k * 90
            # When passing [angle, angle], it means "randomly select an angle within the [angle, angle] interval", 
            # with the interval length being 0. The result is that each rotation is fixed at `angle`
            transforms.append(T.RandomRotation([angle, angle]))

        if cfg.rotation_probability > 0. and self.rotate.value:
            transforms.append(
                T.RandomRotation([self.angle.value, self.angle.value]) # In degree
            )

        if cfg.smooth_probability > 0. and self.smooth.value:
            transforms.append(T.GaussianBlur(kernel_size=3))

        if cfg.contrast_probability > 0. and self.contrast.value:
            transforms.append(T.ColorJitter(contrast=0.5))

        if cfg.resize_probability > 0. and self.resize.value and self.size.value is not None:
            transforms.append(
                _RandomResizedCropSameSize(
                    prob=1.,
                    size=self.size.value,
                )
            )

        if cfg.negate_probability > 0. and self.negate.value:
            transforms.append(T.RandomInvert(p=1.))

        if (
            cfg.roll_probability > 0. and
            self.roll.value and
            self.roll_x.value is not None and
            self.roll_y.value is not None
        ):
            transforms.append(
                _RandomRoll(
                    prob=1.,
                    roll_x=self.roll_x.value,
                    roll_y=self.roll_y.value,
                )
            )

        return T.Compose(transforms)

    def process(
        self,
        images: Union[torch.Tensor, np.ndarray],
        index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        images: [B, C, H, W]

        Return: 
           [B, C, H, W]
        """
        if index is not None:
            return self.children[index].process(images)

        if isinstance(images, torch.Tensor):
            imgs = images
        else:
            imgs = torch.from_numpy(images)

        # BCHW
        self.transform = self._build_transform()
        out = []
        for img in imgs:
            out.append(self.transform(img))

        return torch.stack(out)


@dataclasses.dataclass
class TaskGenerator:
    """Task generator using a dictionary of NumPy arrays as input."""

    def __init__(
        self,
        data: dict[Any, np.ndarray],
        num_labels: int,
        always_same_labels=False,
        use_label_subset: Optional[UseLabelSubset] = None,
    ):
        self.data = data
        self.num_labels = num_labels
        self.always_same_labels = always_same_labels

        if use_label_subset is not None:
            self.use_labels: UseLabelSubset = use_label_subset
        else:
            self.use_labels: UseLabelSubset = list(self.data.keys())

    def _sample_random_labels(
        self,
        labels: list[int],
        batch_size: int,
        same_labels: Optional[bool] = None,
    ) -> LabelGenerator:
        """Generator producing random labels and corr. numbers of samples."""
        if same_labels is None:
            same_labels = self.always_same_labels

        if same_labels:
            chosen_labels = labels[:self.num_labels]
        else:
            chosen_labels = np.random.choice(
                labels, size=self.num_labels, replace=False
            )

        samples_per_label = batch_size // self.num_labels
        labels_with_extra = batch_size % self.num_labels
        for i, label in enumerate(chosen_labels):
            pick = samples_per_label
            if i < labels_with_extra:
                pick += 1
            yield label, pick

    def _images_labels(
        self,
        label_generator: Callable[[], LabelGenerator],
        unlabeled: int = 0,
    ) -> SupervisedSamplesNumpy:
        """Produces labels and images from the label generator."""
        images, labels, classes = [], [], []
        consecutive_label = 0
        for label, num_samples in label_generator():
            sample = self.data[label]
            chosen = np.random.choice(
                range(sample.shape[0]),
                size=num_samples,
                replace=False,
            )
            images.append(sample[chosen, :, :])
            chosen_labels = np.array([consecutive_label] * num_samples)
            remove_label = [(i < unlabeled) for i in range(num_samples)]
            # This indicates that the sample does not have a label.
            chosen_labels[remove_label] = self.num_labels
            classes.append(np.array([label] * num_samples))
            labels.append(chosen_labels)
            consecutive_label += 1
        return images, labels, classes

    def _make_semisupervised_samples(
        self,
        batch_sizes: list[int],
        num_unlabeled_per_class: list[int],
    ) -> list[SupervisedSamplesNumpy]:
        """Helper function for creating multiple semi-supervised samples."""
        output = []
        use_labels = self.use_labels
        if callable(use_labels):
            use_labels = use_labels()
        if not self.always_same_labels:
            # Copying to avoid changing the original list
            use_labels = use_labels[:] # type: ignore
            np.random.shuffle(use_labels)

        for batch_size, unlabeled in zip(batch_sizes, num_unlabeled_per_class):
            # Using the same labelset in all batches.
            label_generator = functools.partial(
                self._sample_random_labels,
                use_labels,
                batch_size,
                same_labels=True,
            )
            output.append(self._images_labels(label_generator, unlabeled))

        return output

    def _make_semisupervised_batches(
        self,
        batch_sizes: list[int],
        num_unlabeled_per_class: list[int],
    ) -> Tuple[np.ndarray, ...]:
        """Creates batches of semi-supervised samples."""
        batches = self._make_semisupervised_samples(
            batch_sizes, num_unlabeled_per_class
        )
        output = []
        for images, labels, classes in batches:
            output.extend([image_mat.astype(np.float32) for image_mat in images])
            output.extend([label_mat.astype(np.int32) for label_mat in labels])
            output.extend([class_mat.astype(np.int32) for class_mat in classes])
        return tuple(output)

    def _make_supervised_batch(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates a batch of supervised samples."""
        batches = self._make_semisupervised_samples([batch_size], [0])
        images, labels, classes = batches[0]
        labels = np.concatenate(labels, axis=0).astype(np.int32)
        images = np.concatenate(images, axis=0).astype(np.float32)
        classes = np.concatenate(classes, axis=0).astype(np.int32)
        return images, labels, classes

    def get_batches(
        self,
        batch_sizes: list[int],
        config: AugmentationConfig,
        num_unlabeled_per_class: list[int],
    ) -> list[SupervisedSample]:
        """Generator producing multiple separate balanced batches of data.

        Arguments:
           batch_sizes: A list of batch sizes for all batches.
           config: Augmentation configuration.
           num_unlabeled_per_class: A list of integers indicating a number of
            "unlabeled" samples per class for each batch.

        Returns:
           A list of (images,labels) pairs produced for each output batch.
           [
              (images_1, labels_1, classes_1),
              (images_2, labels_2, classes_2),
              ...
           ]
        """
        # Returned array is:
        #   output = [
        #       # batch 1
        #       image_1, image_2, ..., image_num_labels,
        #       label_1, label_2, ..., label_num_labels,
        #       class_1, class_2, ..., class_num_labels,
        #
        #       # batch 2
        #       image_1, ..., class_num_labels,
        #       ...
        #   ]
        output = self._make_semisupervised_batches(
            batch_sizes=batch_sizes,
            num_unlabeled_per_class=num_unlabeled_per_class,
        )

        images_labels: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        offs = 0
        for _ in batch_sizes:
            images = output[offs : offs + self.num_labels]
            offs += self.num_labels
            labels = output[offs : offs + self.num_labels]
            offs += self.num_labels
            classes = output[offs : offs + self.num_labels]
            offs += self.num_labels

            images = [torch.from_numpy(img) for img in images]
            labels = [torch.from_numpy(lbl) for lbl in labels]
            classes = [torch.from_numpy(cla) for cla in classes]
            # Processing and combining in batches
            if config.children:
                images = [
                    config.process(image_mat, idx)
                    for idx, image_mat in enumerate(images)
                ]
            else:
                images = [config.process(image_mat) for image_mat in images]
            images_labels.append(
                (
                    torch.cat(images, dim=0), # (batch_size, H, W)
                    torch.cat(labels, dim=0), # (batch_size,)
                    torch.cat(classes, dim=0), # (batch_size,)
                )
            )

        # Shuffling each batch
        output_batches: list[SupervisedSample] = []
        for images, labels, classes in images_labels:
            perm = torch.randperm(images.size(0), device=images.device)
            images = images[perm]
            labels = labels[perm]
            classes = classes[perm]
            output_batches.append((images, labels, classes))

        return output_batches

    def get_batch(
        self,
        batch_size: int,
        config: AugmentationConfig,
        num_unlabeled_per_class: int = 0,
    ) -> SupervisedSample:
        """Generator producing a single batch of data (meta-train + meta-test)."""
        if num_unlabeled_per_class > 0:
            raise ValueError(
                "Unlabeled samples are currently only supported in "
                "balanced inputs."
            )

        images, labels, classes = self._make_supervised_batch(
            batch_size=batch_size
        )
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)
        classes = torch.from_numpy(classes)
        # Augmentation
        images = config.process(images)

        # Shuffle -> Equal to: `tf.range + tf.random.shuffle + tf.gather`
        perm = torch.randperm(images.size(0), device=images.device)
        images = images[perm]
        labels = labels[perm]
        classes = classes[perm]

        return images, labels, classes


class HTDataset(Dataset):
    """
    e.g.
       ```python
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
    ```
    """

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


def make_numpy_data(
    ds: "HTDataset",
    batch_size: int,
    num_labels: int,
    samples_per_label: int,
    image_key: str = "image",
    label_key: str = "label",
    transpose: bool = True,
    max_batches: Optional[int] = None,
) -> dict[int, np.ndarray]:
    """Makes a label-to-samples dictionary from the TF dataset.

    Arguments:
       ds: PyTorch Dataset. Each item should be a mapping containing image and label entries.
        {
            "image": np.ndarray [B, C, H, W],
            "label": np.ndarray [B],
        }
       batch_size: Batch size used by DataLoader.
       num_labels: Total number of labels.
       samples_per_label: Number of samples per label to accumulate.
       image_key: Key of the image tensor.
       label_key: Key of the label tensor.
       transpose: If True, the image is transposed (XY).
       max_batches: If provided, we process no more than this number of batches.

    Returns:
       Dictionary mapping labels to tensors containing all samples (samples_per_label, C, H, W).
    """
    loader: DataLoader[Mapping[str, Any]] = DataLoader(
        ds, batch_size=batch_size, shuffle=False
    )
    examples: dict[int, list[np.ndarray]] = {
        i: [] for i in range(num_labels)
    }
    batch_index: int = 0

    for batch in loader:
        batch_map: Mapping[str, Any] = batch
        images = batch_map[image_key]
        labels = batch_map[label_key]
        # Convert to numpy if needed
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        images_np: np.ndarray = images
        labels_np: np.ndarray = labels

        # Because the size of a Dataset is generally not an integer multiple of the batch size
        samples: int = labels_np.shape[0]
        for idx in range(samples):
            label: int = int(labels_np[idx])
            if len(examples[label]) < samples_per_label:
                examples[label].append(images_np[idx])

        # Stop if enough samples per label are collected
        min_examples: int = min(len(v) for v in examples.values())
        if min_examples >= samples_per_label:
            break

        batch_index += 1
        if max_batches is not None and batch_index >= max_batches:
            break

    """Stack samples per label

    e.g.
       {label_0: [img0, img1, ...],
        label_1: [img0, img1, ...],
        ...}
           ↓
       {label_0: np.ndarray [N, C, H, W],
        label_1: np.ndarray [N, C, H, W],
        ...}
    """
    stacked: dict[int, np.ndarray] = {
        k: np.stack(v, axis=0) for k, v in examples.items()
    }

    # Optional transpose: [N, C, H, W] -> [N, C, W, H]
    if transpose:
        stacked = {
            k: np.transpose(v, axes=(0, 1, 3, 2))
            for k, v in stacked.items()
        }

    return stacked
