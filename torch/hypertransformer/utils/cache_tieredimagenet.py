"""Converts tieredImageNet dataset from pickled files to NumPy."""

import os
import pickle

from absl import app, logging
from absl import flags

import numpy as np

INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path with tieredImageNet pickle files."
)
OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path with tieredImageNet pickle files."
)

# Cache folder for chunked datasets should be named after the dataset
# (and the cache path should point to the parent folder)
SUBFOLDER = "tieredimagenet"


def _images_and_labels(prefix):
    root = INPUT_PATH.value
    imgs = np.load(os.path.join(root, prefix + "_images.npz"))
    lbls = pickle.load(open(os.path.join(root, prefix + "_labels.pkl"), "rb"))
    return imgs["images"], lbls["labels"]


def _collect_images_by_label(images, labels):
    image_dict = {}
    for i in range(len(labels)):
        label = labels[i]
        if label not in image_dict:
            image_dict[label] = []
        image_dict[label].append(images[i])
    return image_dict


def _save_part(chunk_index, tensors):
    tensors = tensors.transpose(0, 1, 4, 2, 3)
    records = len(tensors)
    logging.info(f"Chunk {chunk_index:03d}: {records} records.")
    np.save(
        os.path.join(OUTPUT_PATH.value, SUBFOLDER, f"chunk_{chunk_index:03d}"), tensors,
    )


def collect_and_save(prefix, start_idx, chunk_index=0, num_combine=42):
    """Converts pickled tieredImageNet into a set of sharded numpy arrays."""
    imgs, lbls = _images_and_labels(prefix)
    images = _collect_images_by_label(imgs, lbls)
    labels = []
    tensors = []
    saved_labels = []

    for index, label in enumerate(images):
        label_idx = start_idx + index
        labels.append(label_idx)
        if (index + 1) % num_combine == 0:
            logging.info(f"Saving labels {saved_labels} to chunk {chunk_index}")
            _save_part(chunk_index, tensors)
            chunk_index += 1
            tensors = []
            saved_labels = []
        tensors.append(np.stack(images[label], axis=0))
        saved_labels.append(label_idx)

    if tensors:
        logging.info(f"Saving labels {saved_labels} to chunk {chunk_index}")
        _save_part(chunk_index, tensors)
        chunk_index += 1

    return start_idx + len(images) + 1, labels, chunk_index


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    try:
        os.makedirs(os.path.join(OUTPUT_PATH.value, SUBFOLDER))
    except FileExistsError:
        logging.info("Output folder already exists.")

    label_index = 0
    chunk_index = 0
    labels = {}
    for prefix in ["train", "val", "test"]:
        label_index, labels[prefix], chunk_index = collect_and_save(
            prefix, label_index, chunk_index
        )


if __name__ == "__main__":
    app.run(main)
