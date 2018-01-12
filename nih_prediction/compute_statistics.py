"""Find intensity mean and variance of all NIH Chest Xrays
"""
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from nih_prediction.data import decode_image, read_file


def compute_statistics(nih_paths, verbose=False):
    """Computes the mean and standard deviation of NIH Chest Xray intensities

    Args:
        nih_paths: Paths to all images in NIH Chest Xray data
        verbose: Print progress and final results

    Returns:
        mean, std: Statistics of intensities
    """
    sess = tf.Session()

    batch_size = 100
    dataset = tf.data.Dataset.from_tensor_slices({'path': nih_paths})
    dataset = dataset.map(
        lambda path: decode_image(read_file(path)),
        num_parallel_calls=32
    )
    dataset = dataset.batch(batch_size).prefetch(16)

    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    images = batch['image']

    mean_metric = tf.metrics.mean
    moments, update_ops = list(
        zip(mean_metric(images), mean_metric(images**2))
    )

    sess.run(tf.local_variables_initializer())
    batches = range(len(nih_paths) // batch_size)
    if verbose:
        batches = tqdm.tqdm(batches, unit='batch')
    for _ in batches:
        moment1, moment2 = sess.run(update_ops)
        mean = moment1
        std = np.sqrt(moment2 - moment1**2)
        if verbose:
            batches.set_description('mean: %.3f, std: %.3f' % (mean, std))

    if verbose:
        print('Mean: ', mean)  # 0.49798086
        print('Std: ', std)    # 0.24943127323710504
    return mean, std


def main(nih_path):
    raw_data = pd.read_csv(os.path.join(nih_path, 'Data_Entry_2017.csv'))
    paths = raw_data['Image Index']
    paths = os.path.join(nih_path, 'images/') + paths
    compute_statistics(paths.values, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute statistics of NIH Chest Xray images'
    )
    parser.add_argument('path', help='Path to extracted NIH Chest Xray data')
    args = parser.parse_args()
    main(args.path)
