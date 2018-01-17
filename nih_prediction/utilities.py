"""Utility for towering model across GPUs
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


SLIM_VARIABLES = [slim.model_variable, slim.variable]


def tower(model_fn, batch_input, batch_target, gpus):
    """Creates a towered model with operations on GPUs and Variables on CPU

    The batch is split along the first axis across the GPUs, you must make sure
    the batch is evenly divisible with the number of GPUs.

    Args:
        model_fn: Accepts input and target tensors and returns a loss tensor
        batch_input: Batch of inputs to split across GPUs
        batch_target: Batch of targets to split across GPUs
        gpus: Sequence of GPU index to split across

    Returns:
        Total loss to be optimized (make sure to colocate gradients with ops
        when optimizing)
    """
    batch_size = tf.shape(batch_input)[0]
    batch_slice_size = batch_size // len(gpus)
    losses = list()
    for gpu in gpus:
        i_start = batch_slice_size * gpu
        gpu_slice = slice(i_start, i_start + batch_slice_size)

        # Place operations on a GPU and variables on the CPU
        with tf.device('/gpu:%d' % gpu):
            with tf.name_scope('tower_%d' % gpu) as scope:
                with slim.arg_scope(SLIM_VARIABLES, device='/cpu:0'):
                    losses.append(model_fn(
                        batch_input[gpu_slice],
                        batch_target[gpu_slice]
                    ))

    with tf.device('/cpu:0'):
        return tf.add_n(losses)
