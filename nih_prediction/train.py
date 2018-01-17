"""Train a towered ResNetv2 152 on a number of GPUs
"""
import argparse
import itertools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import tqdm

from nih_prediction.data import build_dataset, get_metadata
from nih_prediction.utilities import tower


MODEL_EXISTS = False

# See https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0  # Avoids bug


def build_model(images, labels):
    """Build ResNet model for multilabel classification

    Variables are placed on CPU and only created once. Global variable
    MODEL_EXISTS keeps track of if these have been created or not.

    Args:
        images: batch of images
        labels: batch of labels

    Returns:
        Loss
    """
    global MODEL_EXISTS
    # Create a new head with sigmoids instead of softmax applied to the logits
    _, resnet = nets.resnet_v2.resnet_v2_152(
        images, num_classes=labels.shape[-1], reuse=MODEL_EXISTS
    )
    MODEL_EXISTS = True
    logits = tf.squeeze(resnet['resnet_v2_152/logits'], axis=[1, 2])
    return tf.losses.sigmoid_cross_entropy(labels, logits)


def train(dataset_train, dataset_val, num_epochs=100, gpus=(0, )):
    """Train the ResNet on the GPUs listed

    Items in datasets should be batches of items as returned by dataset in
    nih_data.build_dataset.

    Args:
        dataset_train: Non repeating dataset for training
        dataset_val: Non repeating dataset for evaluation
        num_epochs: Number of epochs to train the model for
        gpus: Sequence of GPUs to to tower model across
    """
    # Create iterator
    iterator = tf.data.Iterator.from_structure(
        dataset_train.output_types, dataset_train.output_shapes)
    batch = iterator.get_next()

    # Build iterator initializers
    datasets = dict(train=dataset_train, val=dataset_val)
    iterator_init = dict()
    for dataset_name, dataset in datasets.items():
        iterator_init[dataset_name] = iterator.make_initializer(dataset)

    # Build towered model and optimizer
    loss = tower(build_model, batch['image'], batch['label'], gpus)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, colocate_gradients_with_ops=True)

    # Init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train on epochs
    num_batches = dict(train=None, val=None)  # Keep track of number of batches
    for epoch in range(num_epochs):
        for dataset_name in ('train', 'val'):
            print('Epoch %i: %s' % (epoch, dataset_name))
            batch_iter = tqdm.tqdm(
                itertools.count(),
                total=num_batches[dataset_name],
                unit='batch',
                smoothing=1  # No smoothing
            )

            sess.run(iterator_init[dataset_name])
            for i in batch_iter:
                try:
                    batch_loss, _ = sess.run([loss, train_op])
                except tf.errors.OutOfRangeError:
                    num_batches[dataset_name] = i + 1
                    break
                else:
                    batch_iter.set_description('loss: %.3f' % batch_loss)
                break
            batch_iter.close()


def main(args):
    # Load metadata and split to training and validation
    metadata = get_metadata(args.path)
    is_training = np.random.binomial(1, 0.7, len(metadata)).astype(np.bool)
    metas = {
        'train': metadata[is_training],
        'val': metadata[~is_training],
    }

    # Build datasets
    datasets = dict()
    for name, meta in metas.items():
        dataset = build_dataset(
            meta,
            mean=np.float32(args.mean) if args.mean is not None else None,
            std=np.float32(args.std) if args.std is not None else None,
            num_parallel_calls=args.num_workers
        )
        datasets[name] = dataset.shuffle(1000).batch(args.batch_size)

    # Run training
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    train(datasets['train'], datasets['val'], gpus=gpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ResNet152 v2 to predict labels of NIH Chest Xrays'
    )
    parser.add_argument('--num-workers', default=1, type=int,
                        help='Number of data reading workers')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--mean', type=float, default=0.49798086,
                        help='Mean intensity of scans for normalization')
    parser.add_argument('--std', type=float, default=0.24943127323710504,
                        help='Std of intensity of scans for normalization')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma separated list of GPUs to use')
    parser.add_argument('path', help='Path to extracted NIH Chest Xray data')
    args = parser.parse_args()
    main(args)
