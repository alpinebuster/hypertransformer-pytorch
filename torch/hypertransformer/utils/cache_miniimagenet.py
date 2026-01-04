"""Converts miniimagenet dataset from pickled files to NumPy."""

import dataclasses
import os
import pickle

from typing import Any

from absl import app
from absl import flags

import numpy as np

INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path with miniImageNet pickle files."
)
OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path with miniImageNet pickle files."
)


@dataclasses.dataclass
class Sources:
    data: dict[Any, Any] = dataclasses.field(default_factory=dict)


def pickle_path(root, split):
    path = os.path.join(root, f"mini-imagenet-cache-{split}.pkl")
    if not os.path.exists(path):
        raise RuntimeError(f"Pickle file {path} is not found!")
    return path


def get_data(root):
    data = {
        split: pickle.loads(open(pickle_path(root, split), "rb").read())
        for split in ["train", "test", "val"]
    }
    return Sources(data=data)


def get_combined(data):
    outputs = []
    for split in ["train", "val", "test"]:
        classes = data.data[split]["class_dict"]
        images = data.data[split]["image_data"]
        for values in classes.values():
            from_class = np.min(values)
            to_class = np.max(values) + 1
            outputs.append(images[from_class:to_class])
    return np.stack(outputs, axis=0)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    data = get_data(INPUT_PATH.value)
    combined = get_combined(data)
    combined_ch_first = combined.transpose(0, 1, 4, 2, 3)
    assert combined_ch_first.shape == (100, 600, 3, 84, 84)
    try:
        os.makedirs(OUTPUT_PATH.value, exist_ok=True)
    finally:
        np.save(os.path.join(OUTPUT_PATH.value, "miniimagenet"), combined_ch_first)


if __name__ == "__main__":
    app.run(main)
