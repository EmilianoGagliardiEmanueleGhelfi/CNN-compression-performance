"""
This script takes in input a pb file, restores the weights of the network, and then computes the ranges for each layer
"""
import argparse
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.contrib.util import make_ndarray


def get_weights_from_pb(pb_file):
    """
    takes in input a pb file, restores the weights of the network, and returns a list containing a ndarray with the
    weights and the biases for each layer
    :param pb_file: path to the pb file of the network
    :return: a list containing for each layer a ndarray with its parameters
    """
    weights_to_ret = []
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # get all the weights tensor as np.array
        weights = [make_ndarray(node_def.attr['value'].tensor).flatten().tolist()
                   for node_def in graph_def.node
                   if node_def.attr['value'].tensor.dtype is not 0 and ('Variable' in node_def.name or 'kernel' in node_def.name or 'bias' in node_def.name or 'weights' in node_def.name)]
        # flatten the elements in weights
        for i in range(0,len(weights),2):
            weights_to_ret.append(weights[i]+weights[i+1])
        return weights_to_ret


def get_ranges(weights):
    """

    :param weights: a list containing the parameters for each layer
    :return: a list of tuple (min, max) for each layer
    """
    return [(min(params), max(params)) for params in weights]


help = 'This script takes in input a pb file, restores the weights of the network, and then computes the ranges for ' \
       'each layer '

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--pb_file', type=str, nargs=1, help='pb file containing a neural network')
    args = parser.parse_args()
    weights = get_weights_from_pb(args.pb_file[0])
    print get_ranges(weights)
