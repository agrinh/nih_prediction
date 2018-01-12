"""Produce metadata and datasets of NIH Chest Xray images
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf


def get_metadata(path):
    """Produce metadata with relevant columns from NIH Chest Xray images

    Args:
        path: Path to NIH dataset

    Returns:
        metadata Dataframe with image and label
    """
    raw_meta = pd.read_csv(os.path.join(path, 'Data_Entry_2017.csv'))
    meta = raw_meta[['Image Index', 'Finding Labels']].copy()
    meta.columns = ['image', 'label']
    meta.image = os.path.join(path, 'images/') + meta.image
    return meta


def build_dataset(meta, mean=None, std=None, num_parallel_calls=32):
    """Produce tf Dataset from metadata

    If mean and std are provided those values will be used to normalise the
    image intensities to zero mean and unit variance.

    Args:
        meta: Dataframe with paths to images under column name image
        mean:
        std: If both provided will be used to normalize images
        num_parallel_calls: Number of threads for loading images
    """
    encoded_labels = meta.label.str.get_dummies(sep='|').sort_index(axis=1)
    ds = tf.data.Dataset.from_tensor_slices({
        'index': meta.index,
        'path': meta['image'].values,
        'label': encoded_labels.values.astype(np.float32)
    })
    if None in (mean, std):
        mean = 0
        std = 1
    return ds.map(
        lambda item: normalize_image(decode_image(read_file(item)), mean, std),
        num_parallel_calls=num_parallel_calls
    )


def read_file(item):
    """Read file in key path into key image
    """
    item['image'] = tf.read_file(item['path'])
    return item


def decode_image(item):
    """Decode raw image file into float32 image tensor with key image
    """
    decoded = tf.image.decode_image(item['image'])
    item['image'] = tf.image.convert_image_dtype(decoded, tf.float32)
    # All images are B&W, but some seem to have the channel replicated,
    # to avoid issues we simply select the first channel
    item['image'] = tf.expand_dims(item['image'][:, :, 0], axis=-1)
    item['image'].set_shape([None, None, 1])
    return item


def normalize_image(item, mean, std):
    """Normalize image with key image to zero mean and unit variance
    """
    item['image'] = (item['image'] - mean) / std
    return item
