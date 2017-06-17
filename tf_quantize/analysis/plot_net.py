"""
For the net_performance files in input plots:
- size of the original and the quantized network
- accuracy of the original and the quantized
- cache misses of the original and the quantized (L1-dcache-load-misses)
- test time
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import json
import numpy as np
from matplotlib import mlab
import os
import math
from os import listdir
from os.path import isfile, join
import math
import weights_ranges
from tf_quantize.net_perf.net_performance import NetPerformance

help = 'This script takes in input performance files and plot comparisons between original and quantized network'


def plot(net_files):
    if not os.path.exists('img'):
        os.mkdir('img')
    # deserialize file
    net_list = []
    for net_file in net_files:
        with open(net_file, 'r') as data_file:
            nets = json.load(data_file)
        net1 = NetPerformance(json_dict=nets[0])
        net2 = NetPerformance(json_dict=nets[1])
        net_list.append(net1)
        net_list.append(net2)

    # accuracy
    accuracy_list = [n.accuracy for n in net_list]
    original_acc = [accuracy_list[i] for i in range(0, len(accuracy_list), 2)]
    quantized_acc = [accuracy_list[i] for i in range(1, len(accuracy_list), 2)]
    bar_chart(original_acc, quantized_acc, (net_list[i].name for i in range(0, len(net_list),2)), "Accuracy",
              "Comparison of accuracies", "img/acc.png")

    # file size
    file_size = [n.size for n in net_list]
    original_size = [file_size[i] for i in range(0, len(file_size), 2)]
    quantized_size = [file_size[i] for i in range(1, len(file_size), 2)]
    bar_chart(original_size, quantized_size, (net_list[i].name for i in range(0, len(net_list), 2)), "File Size",
              "Comparison of size", "img/size.png")

    # l1 d cache load misses
    misses = [n.L1_dcache_load_misses_mean for n in net_list]
    misses_std = [math.sqrt(n.L1_dcache_load_misses_var) for n in net_list]
    original_misses = [misses[i] for i in range(0, len(misses), 2)]
    quantized_misses = [misses[i] for i in range(1, len(misses), 2)]
    original_misses_std = [misses_std[i] for i in range(0, len(misses), 2)]
    quantized_misses_std = [misses_std[i] for i in range(1, len(misses), 2)]
    bar_chart(original_misses, quantized_misses, (net_list[i].name for i in range(0, len(net_list), 2)), "Misses",
              "Comparison of l1 dcache load misses", "img/misses.png", original_std=original_misses_std, quant_std=quantized_misses_std)

    # test time
    test_time = [n.test_time_mean for n in net_list]
    test_time_std = [math.sqrt(n.test_time_var) for n in net_list]
    original_time = [test_time[i] for i in range(0, len(test_time), 2)]
    quantized_time = [test_time[i] for i in range(1, len(test_time), 2)]
    original_time_std = [test_time_std[i] for i in range(0, len(test_time), 2)]
    quantized_time_std = [test_time_std[i] for i in range(1, len(test_time), 2)]
    bar_chart(original_time, quantized_time, (net_list[i].name for i in range(0, len(net_list), 2)), "Inference Time",
              "Comparison of Inference time", "img/test_time.png",original_std=original_time_std, quant_std=quantized_time_std)

    # cache misses
    cache_misses = [n.cache_misses_mean for n in net_list]
    cache_misses_std = [math.sqrt(n.cache_misses_var) for n in net_list]
    original_cache_misses = [cache_misses[i] for i in range(0, len(test_time), 2)]
    quantized_cache_misses = [cache_misses[i] for i in range(1, len(test_time), 2)]
    original_cache_misses_std = [cache_misses_std[i] for i in range(0, len(test_time), 2)]
    quantized_cache_misses_std = [cache_misses_std[i] for i in range(1, len(test_time), 2)]
    bar_chart(original_cache_misses, quantized_cache_misses, (net_list[i].name for i in range(0, len(net_list), 2)), "Cache Misses",
              "Comparison of cache misses", "img/cache_misses.png", original_std=original_cache_misses_std,
              quant_std=quantized_cache_misses_std)

    # plot weights
    net_list_weights = [net_list[i] for i in range(0,len(net_list),2)]
    for net in net_list_weights:
        weights = weights_ranges.get_weights_from_pb(net.path)
        histogram(weights, 'img/weights'+net.name+'.png', net.name)


def bar_chart(original_data, quantized_data, xNames, yLabel, title, filename, original_std = None, quant_std = None):
    """
    :param original_data: original data to plot
    :param quantized_data: quantized data to plot
    :param xNames: names on the x axes
    :param yLabel: Label of y axis
    :param title: title of the plot
    :param filename: filename to write
    :param original_std: std deviation of original data
    :param quant_std: std deviation of quantized data
    :return:
    """
    # length of the array, to get the correct indexes
    N = len(original_data)
    # index of elements
    x = np.arange(N)
    # width of the bar
    width = 0.2
    if original_std is None:
        rects1 = plt.bar(x, original_data, width,
                         color='b',
                         label='Original')
        rects2 = plt.bar(x + width, quantized_data, width,
                         color='r',
                         label='Quantized')
    else:
        rects1 = plt.bar(x, original_data, width,
                         color='b',
                         label='Original', yerr=original_std)
        rects2 = plt.bar(x + width, quantized_data, width,
                         color='r',
                         label='Quantized', yerr=quant_std)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.xticks(x + width, xNames, rotation = 45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def histogram(weights_list, filename, net_name):
    fig, axes = plt.subplots(nrows=int(math.ceil(float(len(weights_list))/float(2))), ncols=2)
    ax_list = axes.flatten()
    for i in range(0,len(weights_list)):
        num_bins = 100
        n, bins, patches = ax_list[i].hist(weights_list[i], num_bins)
        ax_list[i].set_title('Layer '+str(i))
    #fig.suptitle('Weights distribution for ' + net_name)
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--net_file', type=str, nargs='+', help='files created by workflow program')
    parser.add_argument('--dir',type=str,nargs=1,help='directory containing file')
    args = parser.parse_args()
    if len(args.dir) == 0:
        plot(args.net_file)
    else:
        files = [args.dir[0]+'/'+f for f in listdir(args.dir[0]) if isfile(join(args.dir[0], f))]
        plot(files)
