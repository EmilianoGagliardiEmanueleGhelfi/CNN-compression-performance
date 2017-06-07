"""
This script takes in input a pb file, restores the weights of the network, and then computes the ranges for each layer
"""
import argparse

import numpy
import tensorflow as tf
from tensorflow.contrib.util import make_ndarray

from tf_quantize.weights_analysis.plot_weights import vis_square


def weights_ranges(pb_file):
    """
    takes in input a pb file, restores the weights of the network, and then computes the ranges for each layer
    :param pb_file: paht to the pb file of the network
    :return: a list containing for each layer with variables a tuple (min, max)
    """
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # get all the weights tensor as np.array
        weights = [make_ndarray(node_def.attr['value'].tensor)
                   for node_def in graph_def.node
                   if node_def.attr['value'].tensor.dtype is not 0 and 'Variable' in node_def.name]
    # now weights contains a sequence like: weights1, bias1, weights2, bias2...
    ranges = []
    # find the max and the min in the set composed by bias and weights for each layer
    for i in range(0, len(weights), 2):
        max_w = weights[i].max()
        max_b = weights[i + 1].max()
        min_w = weights[i].min()
        min_b = weights[i + 1].min()
        max_bw = max([max_w, max_b])
        min_bw = min([min_w, min_b])
        ranges.append((min_bw, max_bw))
    return ranges


help = 'This script takes in input a pb file, restores the weights of the network, and then computes the ranges for ' \
       'each layer '

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--pb_file', type=str, nargs=1, help='pb file containing a neural network')
    args = parser.parse_args()
    print weights_ranges(args.pb_file[0])
