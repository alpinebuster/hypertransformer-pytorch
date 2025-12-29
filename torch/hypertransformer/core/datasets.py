"""Task generator using basic datasets."""

import dataclasses
import functools

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.ops import roi_align
import torchvision.transforms.functional as TV_F

from hypertransformer.core import common_ht

DatasetInfo = common_ht.DatasetInfo

UseLabelSubset = Union[List[int], Callable[[], List[int]]]
LabelGenerator = Generator[Tuple[int, int], None, None]
SupervisedSamples = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
DSBatch = Tuple[torch.Tensor, torch.Tensor]


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
    target_size: Union[int, Tuple[int, int], List[int]],
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
        # (C,H,W) → (1,C,H,W)
        images = images.unsqueeze(0)
        # bbox (4,) → (1,4)
        bboxes = bboxes.unsqueeze(0)
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
    target_size: Union[int, Tuple[int, int], List[int]],
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
        out = TV_F.hflip(image_tensor) if torch.rand(()) < 0.5 else image_tensor
        # hue_factor=`2*torch.rand(())-1` ∈ [-1, 1) -> [-hue_delta, +hue_delta]
        out = TV_F.adjust_hue(out, hue_factor=hue_delta * (2*torch.rand(size=()).item() - 1))
        # brightness_factor ∈ [1 - brightness_delta, 1 + brightness_delta]
        out = TV_F.adjust_brightness(
            out, brightness_factor=1.0 + brightness_delta * (2*torch.rand(size=()).item() - 1)
        )
        # contrast_factor ∈ [contrast[0], contrast[1]]
        out = TV_F.adjust_contrast(
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


class _RandomValue:
    """Random value."""

    value: Optional[tf.Variable] = None

    def create(self, name, dtype=tf.bool):
        """Create a random value of a given type."""
        if self.value is None:
            self.value = tf.get_variable(name, shape=(), dtype=dtype, trainable=False)

    def _random_bool(self, prob):
        return tf.math.less(tf.random.uniform(shape=(), maxval=1.0), prob)

    def assign_bool(self, prob):
        """Operator assigning a random boolean value to `value`."""
        return tf.assign(self.value, self._random_bool(prob))

    def assign_uniform(self, scale):
        """Operator assigning a random uniform value to `value`."""
        rand = tf.random.uniform(shape=(), minval=-1.0, maxval=1.0) * scale
        return tf.assign(self.value, rand)

@dataclasses.dataclass
class AugmentationConfig:
    """Configuration of the image augmentation generator."""

    random_config: Optional[RandomizedAugmentationConfig] = None

    rotate:   _RandomValue = dataclasses.field(default_factory=_RandomValue)
    smooth:   _RandomValue = dataclasses.field(default_factory=_RandomValue)
    contrast: _RandomValue = dataclasses.field(default_factory=_RandomValue)
    negate:   _RandomValue = dataclasses.field(default_factory=_RandomValue)
    roll:     _RandomValue = dataclasses.field(default_factory=_RandomValue)
    resize:   _RandomValue = dataclasses.field(default_factory=_RandomValue)
    angle:    _RandomValue = dataclasses.field(default_factory=_RandomValue)
    alpha:    _RandomValue = dataclasses.field(default_factory=_RandomValue)
    size:     _RandomValue = dataclasses.field(default_factory=_RandomValue)
    roll_x:   _RandomValue = dataclasses.field(default_factory=_RandomValue)
    roll_y:   _RandomValue = dataclasses.field(default_factory=_RandomValue)
    rotate_90_times: _RandomValue = dataclasses.field(default_factory=_RandomValue)

    children: List["AugmentationConfig"] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.children:
            return

        for name in ["rotate", "smooth", "contrast", "negate", "resize", "roll"]:
            getattr(self, name).create(name)
        for name in [
            "angle",
            "alpha",
            "size",
            "roll_x",
            "roll_y",
            "rotate_90_times",
        ]:
            getattr(self, name).create(name, tf.float32)

    def randomize_op(self):
        """Randomizes the augmentation according to the `random_config`."""
        if self.children:
            return tf.group([child.randomize_op() for child in self.children])
        if self.random_config is None:
            return tf.no_op()

        config = self.random_config
        assign_rotate = self.rotate.assign_bool(config.rotation_probability)
        assign_smooth = self.smooth.assign_bool(config.smooth_probability)
        assign_contrast = self.contrast.assign_bool(config.contrast_probability)
        assign_negate = self.negate.assign_bool(config.negate_probability)
        assign_resize = self.resize.assign_bool(config.resize_probability)
        assign_roll = self.roll.assign_bool(config.roll_probability)
        angle_range = config.angle_range / 180.0 * np.pi
        assign_angle = self.angle.assign_uniform(scale=angle_range)
        assign_size = self.size.assign_uniform(scale=1.0)
        assign_alpha = self.alpha.assign_uniform(scale=1.0)
        assign_roll_x = self.roll_x.assign_uniform(scale=config.roll_range)
        assign_roll_y = self.roll_y.assign_uniform(scale=config.roll_range)
        assign_rotate_90 = self.rotate_90_times.assign_uniform(scale=2.0)
        return tf.group(
            assign_rotate,
            assign_smooth,
            assign_contrast,
            assign_angle,
            assign_negate,
            assign_resize,
            assign_alpha,
            assign_size,
            assign_roll,
            assign_roll_x,
            assign_roll_y,
            assign_rotate_90,
        )

    def _normalize(self, image):
        v_min = tf.reduce_min(image, axis=(1, 2), keepdims=True)
        v_max = tf.reduce_max(image, axis=(1, 2), keepdims=True)
        return (image - v_min) / (v_max - v_min + 1e-7)

    def _aug_contrast(self, image):
        """Increases the image contrast."""
        normalized = image / 128.0 - 1
        v_mean = tf.reduce_mean(normalized, axis=(1, 2), keepdims=True)
        mult, shift = (ALPHA_MAX - ALPHA_MIN) / 2, (ALPHA_MAX + ALPHA_MIN) / 2
        alpha = mult * self.alpha.value + shift
        output = tf.math.tanh((normalized - v_mean) * alpha) / alpha
        output += v_mean
        return 255.0 * self._normalize(output)

    def _aug_smooth(self, images):
        """Smooths the image using a 5x5 uniform kernel."""
        depth = int(images.shape[-1])
        image_filter = tf.eye(num_rows=depth, num_columns=depth, dtype=tf.float32)
        image_filter = tf.stack([image_filter] * 3, axis=0)
        image_filter = tf.stack([image_filter] * 3, axis=0)
        output = tf.nn.conv2d(images, image_filter / 9.0, padding="SAME")
        return 255.0 * self._normalize(output)

    def _aug_roll(self, images):
        """Smooths the image using a 5x5 uniform kernel."""
        width, height = images.shape[1], images.shape[2]
        width, height = tf.cast(width, tf.float32), tf.cast(height, tf.float32)
        x = tf.cast(self.roll_x.value * width, tf.int32)
        y = tf.cast(self.roll_y.value * height, tf.int32)
        return tf.roll(images, [x, y], axis=[1, 2])

    def _aug_negate(self, images):
        """Negates the image."""
        return 255.0 - images

    def _aug_resize(self, images):
        """Resizes the image."""
        size = int(images.shape[1])
        re_size = size * (1 - MAX_RESIZE / 2 + MAX_RESIZE * self.size.value / 2)
        images = tf.image.resize(images, [re_size, re_size])
        return tf.image.resize_with_crop_or_pad(images, size, size)

    def _aug_rotate_90(self, images):
        num_rotations = tf.cast(
            tf.math.floor(self.rotate_90_times.value + 2.0), tf.int32
        )
        return tf.image.rot90(images, k=num_rotations)

    def _aug_rotate(self, images, angles, fill_value=0.0):
        """
        images: Tensor, shape [B, H, W, C], dtype float32 (or castable)
        angles: scalar Tensor (radians) or Tensor of shape [B], dtype float32
        fill_value: scalar to use for pixels sampled outside source image

        returns: rotated images, same shape as input
        """
        images = tf.cast(images, tf.float32)
        shape = tf.shape(images)
        batch = shape[0]
        H = shape[1]
        W = shape[2]
        C = shape[3]

        # Ensure angles is shape [B]
        angles = tf.convert_to_tensor(angles, dtype=tf.float32)
        angles = tf.reshape(angles, [-1])
        angles = tf.cond(
            tf.equal(tf.size(angles), 1),
            lambda: tf.tile(angles, [batch]),
            lambda: angles,
        )
        angles = tf.reshape(angles, [batch])

        # Grid of (x, y) coordinates in the output image
        x_lin = tf.cast(
            tf.linspace(0.0, tf.cast(W, tf.float32) - 1.0, W), tf.float32
        )  # [W]
        y_lin = tf.cast(
            tf.linspace(0.0, tf.cast(H, tf.float32) - 1.0, H), tf.float32
        )  # [H]
        x_t, y_t = tf.meshgrid(x_lin, y_lin)  # [H, W]
        x_flat = tf.reshape(x_t, [-1])  # [HW]
        y_flat = tf.reshape(y_t, [-1])  # [HW]
        HW = tf.shape(x_flat)[0]

        # Coordinates relative to image center
        cx = (tf.cast(W, tf.float32) - 1.0) / 2.0
        cy = (tf.cast(H, tf.float32) - 1.0) / 2.0
        x_rel = x_flat - cx  # [HW]
        y_rel = y_flat - cy  # [HW]

        # Above line is to satisfy graph shape inference; replace with tile:
        x_rel = tf.tile(tf.expand_dims(x_rel, 0), [batch, 1])  # [B, HW]
        y_rel = tf.tile(tf.expand_dims(y_rel, 0), [batch, 1])  # [B, HW]

        # cos & sin per-batch, shape [B,1]
        cos_a = tf.reshape(tf.cos(angles), [batch, 1])
        sin_a = tf.reshape(tf.sin(angles), [batch, 1])

        # Inverse mapping: source_coord = R_{-theta} * (out_coord - center) + center
        # R_{-theta} = [[cos, sin], [-sin, cos]]
        x_src = cos_a * x_rel + sin_a * y_rel + cx  # [B, HW]
        y_src = -sin_a * x_rel + cos_a * y_rel + cy  # [B, HW]

        # Bilinear interpolation
        x0 = tf.floor(x_src)
        x1 = x0 + 1.0
        y0 = tf.floor(y_src)
        y1 = y0 + 1.0

        x0_safe = tf.cast(x0, tf.int32)
        x1_safe = tf.cast(x1, tf.int32)
        y0_safe = tf.cast(y0, tf.int32)
        y1_safe = tf.cast(y1, tf.int32)

        x0_clipped = tf.clip_by_value(x0_safe, 0, W - 1)
        x1_clipped = tf.clip_by_value(x1_safe, 0, W - 1)
        y0_clipped = tf.clip_by_value(y0_safe, 0, H - 1)
        y1_clipped = tf.clip_by_value(y1_safe, 0, H - 1)

        # Weights
        wa = (x1 - x_src) * (y1 - y_src)  # top-left
        wb = (x1 - x_src) * (x_src - x0)  # bottom-left
        wc = (x_src - x0) * (y1 - y_src)  # top-right
        wd = (x_src - x0) * (x_src - x0)  # bottom-right

        # Correct bilinear weights: (x1 - x)*(y1 - y), (x1 - x)*(y - y0), (x - x0)*(y1 - y), (x - x0)*(y - y0)
        wa = (x1 - x_src) * (y1 - y_src)
        wb = (x1 - x_src) * (y_src - y0)
        wc = (x_src - x0) * (y1 - y_src)
        wd = (x_src - x0) * (y_src - y0)

        # Build indices for gather_nd: shape [B*HW, 3] where each row is [b, y, x]
        batch_idx = tf.reshape(tf.range(batch, dtype=tf.int32), [batch, 1])  # [B,1]
        batch_idx = tf.tile(batch_idx, [1, HW])  # [B, HW]

        def gather_at(x_inds, y_inds):
            # x_inds, y_inds: [B, HW] int32
            idx = tf.stack(
                [
                    tf.reshape(batch_idx, [-1]),
                    tf.reshape(y_inds, [-1]),
                    tf.reshape(x_inds, [-1]),
                ],
                axis=1,
            )  # [B*HW, 3]
            vals = tf.gather_nd(images, idx)  # [B*HW, C]
            vals = tf.reshape(vals, [batch, HW, C])  # [B, HW, C]
            return vals

        Ia = gather_at(x0_clipped, y0_clipped)  # top-left
        Ib = gather_at(x0_clipped, y1_clipped)  # bottom-left
        Ic = gather_at(x1_clipped, y0_clipped)  # top-right
        Id = gather_at(x1_clipped, y1_clipped)  # bottom-right

        # Weights shapes: [B, HW] -> make [B, HW, 1]
        wa = tf.expand_dims(wa, -1)
        wb = tf.expand_dims(wb, -1)
        wc = tf.expand_dims(wc, -1)
        wd = tf.expand_dims(wd, -1)

        out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # [B, HW, C]

        # Mask for points that were outside original image bounds (so we can fill with fill_value)
        inside_x = tf.logical_and(x_src >= 0.0, x_src <= tf.cast(W, tf.float32) - 1.0)
        inside_y = tf.logical_and(y_src >= 0.0, y_src <= tf.cast(H, tf.float32) - 1.0)
        inside = tf.cast(tf.logical_and(inside_x, inside_y), tf.float32)  # [B, HW]
        inside = tf.expand_dims(inside, -1)  # [B, HW, 1]

        out = out * inside + fill_value * (1.0 - inside)
        out = tf.reshape(out, [batch, H, W, C])
        return out

    def process(self, images, index=None):
        """Processes a batch of samples."""
        if index is not None:
            return self.children[index].process(images)
        config = self.random_config
        if config is None:
            raise ValueError("AugmentationConfig is undefined.")
        if config.rotate_by_90:
            images = self._aug_rotate_90(images)
        if config.rotation_probability > 0.0:
            images = tf.cond(
                self.rotate.value,  # radian
                lambda: self._aug_rotate(images, self.angle.value),
                lambda: tf.identity(images),
            )
        if config.roll_probability > 0.0:
            images = tf.cond(
                self.roll.value,
                lambda: self._aug_roll(images),
                lambda: tf.identity(images),
            )
        if config.resize_probability > 0.0:
            images = tf.cond(
                self.resize.value,
                lambda: self._aug_resize(images),
                lambda: tf.identity(images),
            )
        if config.contrast_probability > 0.0:
            images = tf.cond(
                self.contrast.value,
                lambda: self._aug_contrast(images),
                lambda: tf.identity(images),
            )
        if config.smooth_probability > 0.0:
            images = tf.cond(
                self.smooth.value,
                lambda: self._aug_smooth(images),
                lambda: tf.identity(images),
            )
        if config.negate_probability > 0.0:
            images = tf.cond(
                self.negate.value,
                lambda: self._aug_negate(images),
                lambda: tf.identity(images),
            )
        return images


@dataclasses.dataclass
class TaskGenerator:
    """Task generator using a dictionary of NumPy arrays as input."""

    def __init__(
        self,
        data: Dict[Any, np.ndarray],
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

    def _labels_per_batch(self, batch_size: int) -> list[int]:
        samples_per_label = batch_size // self.num_labels
        labels_with_extra = batch_size % self.num_labels
        output = []
        for i in range(self.num_labels):
            if i < labels_with_extra:
                output.append(samples_per_label + 1)
            else:
                output.append(samples_per_label)
        return output

    def sample_random_labels(
        self,
        labels: list[int],
        batch_size: int,
        same_labels: Optional[bool] = None,
    ) -> LabelGenerator:
        """Generator producing random labels and corr. numbers of samples."""
        if same_labels is None:
            same_labels = self.always_same_labels

        if same_labels:
            chosen_labels = labels[: self.num_labels]
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
    ) -> SupervisedSamples:
        """Produces labels and images from the label generator."""
        images, labels, classes = [], [], []
        consecutive_label = 0
        for label, num_samples in label_generator():
            sample = self.data[label]
            chosen = np.random.choice(
                range(sample.shape[0]), size=num_samples, replace=False
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
        self, batch_sizes: list[int], num_unlabeled_per_class: list[int]
    ) -> list[SupervisedSamples]:
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
                self.sample_random_labels, use_labels, batch_size, same_labels=True
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
    ) -> list[SupervisedSamples]:
        """Generator producing multiple separate balanced batches of data.

        Arguments:
          batch_sizes: A list of batch sizes for all batches.
          config: Augmentation configuration.
          num_unlabeled_per_class: A list of integers indicating a number of
            "unlabeled" samples per class for each batch.

        Returns:
          A list of (images,labels) pairs produced for each output batch.
        """
        sup_sample = functools.partial(
            self._make_semisupervised_batches,
            num_unlabeled_per_class=num_unlabeled_per_class,
            batch_sizes=batch_sizes,
        )
        #                   [   |------------------------------------------batch 1--------------------------------|        , ...]
        # Returned array is [image-1, ..., image-num_labels, label-1, ..., label-num_labels, class-1, ..., class-num_labels, ...]
        types = [tf.float32] * self.num_labels
        types += [tf.int32] * self.num_labels
        types += [tf.int32] * self.num_labels
        types = types * len(batch_sizes)
        output = tf.py_func(sup_sample, [], types, stateful=True)

        images_labels = []
        some_label = list(self.data.keys())[0]
        offs = 0
        for batch_size in batch_sizes:
            images = output[offs : offs + self.num_labels]
            offs += self.num_labels
            labels = output[offs : offs + self.num_labels]
            offs += self.num_labels
            classes = output[offs : offs + self.num_labels]
            offs += self.num_labels
            # Setting a proper shape for post-processing to work
            samples_per_label = self._labels_per_batch(batch_size)
            for image, num_samples in zip(images, samples_per_label):
                image.set_shape([num_samples] + list(self.data[some_label][0].shape))
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
                    tf.concat(images, axis=0),
                    tf.concat(labels, axis=0),
                    tf.concat(classes, axis=0),
                )
            )

        # Shuffling each batch
        output: list[SupervisedSamples] = []
        for images, labels, classes in images_labels:
            indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            images = tf.gather(images, shuffled_indices)
            labels = tf.gather(labels, shuffled_indices)
            classes = tf.gather(classes, shuffled_indices)
            output.append((images, labels, classes))
        return output

    def get_batch(
        self,
        batch_size: int,
        config: AugmentationConfig,
        num_unlabeled_per_class: int = 0,
    ) -> SupervisedSamples:
        """Generator producing a single batch of data (meta-train + meta-test)."""
        if num_unlabeled_per_class > 0:
            raise ValueError(
                "Unlabeled samples are currently only supported in " "balanced inputs."
            )
        sup_sample = functools.partial(
            self._make_supervised_batch, batch_size=batch_size
        )
        images, labels, classes = tf.py_func(
            sup_sample, [], (tf.float32, tf.int32, tf.int32), stateful=True
        )
        some_label = list(self.data.keys())[0]
        # Setting a proper shape for post-processing to work
        images.set_shape([batch_size] + list(self.data[some_label][0].shape))
        images = config.process(images)

        indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        images = tf.gather(images, shuffled_indices)
        labels = tf.gather(labels, shuffled_indices)
        classes = tf.gather(classes, shuffled_indices)

        return images, labels, classes


def make_numpy_data(
    sess,
    ds: Dataset,
    batch_size: int,
    num_labels: int,
    samples_per_label: int,
    image_key="image",
    label_key="label",
    transpose=True,
    max_batches=None,
) -> Dict[int, np.ndarray]:
    """Makes a label-to-samples dictionary from the TF dataset.

    Arguments:
      sess: Initialized TF session.
      ds: Torch dataset.
      batch_size: batch size to use for processing data.
      num_labels: total number of labels.
      samples_per_label: number of samples per label to accumulate.
      image_key: key of the image tensor.
      label_key: key of the label tensor.
      transpose: if True, the image is transposed (XY).
      max_batches: if provided, we process no more than this number of batches.

    Returns:
      Dictionary mapping labels to tensors containing all samples.
    """
    data = tf.data.make_one_shot_iterator(ds.batch(batch_size)).get_next()

    examples = {i: [] for i in range(num_labels)}
    batch_index = 0
    while True:
        """
        {
            "image": np.ndarray [B, H, W, C],
            "label": np.ndarray [B],
        }
        """
        value = sess.run(data)
        samples = value[label_key].shape[0]
        for index in range(samples):
            label = int(value[label_key][index])
            if len(examples[label]) < samples_per_label:
                examples[label].append(value[image_key][index])
        # Checking if we accumulated enough.
        min_examples = np.min([len(examples[key]) for key in examples])
        if min_examples >= samples_per_label:
            break
        # Checking if we already processed too many batches.
        batch_index += 1
        if max_batches is not None and batch_index >= max_batches:
            break

    examples = {k: np.stack(v, axis=0) for k, v in examples.items()}
    if transpose:
        examples = {k: np.transpose(v, axes=[0, 2, 1, 3]) for k, v in examples.items()}
    return examples
