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

from tf_quantize.net_perf.net_performance import NetPerformance

help = 'This script takes in input performance files and plot comparisons between original and quantized network'


def plot(net_files):
    print net_files
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
    misses = [n.L1_dcache_load_misses for n in net_list]
    original_misses = [misses[i] for i in range(0, len(misses), 2)]
    quantized_misses = [misses[i] for i in range(1, len(misses), 2)]
    bar_chart(original_misses, quantized_misses, (net_list[i].name for i in range(0, len(net_list), 2)), "Misses",
              "Comparison of l1 dcache load misses", "img/misses.png")
    # test time
    test_time = [n.test_time for n in net_list]
    original_time = [test_time[i] for i in range(0, len(test_time), 2)]
    quantized_time = [test_time[i] for i in range(1, len(test_time), 2)]
    bar_chart(original_time, quantized_time, (net_list[i].name for i in range(0, len(net_list), 2)), "Inference Time",
              "Comparison of Inference time", "img/test_time.png")
    # plot weights


def bar_chart(original_data, quantized_data, xNames, yLabel, title,filename):
    """
    :param original_data: original data to plot
    :param quantized_data: quantized data to plot
    :param xNames: names on the x axes
    :param yLabel: Label of y axis
    :param title: title of the plot
    :param filename: filename to write
    :return:
    """
    # length of the array, to get the correct indexes
    N = len(original_data)
    # index of elements
    x = np.arange(N)
    # width of the bar
    width = 0.2
    rects1 = plt.bar(x, original_data, width,
                     color='b',
                     label='Original')
    rects2 = plt.bar(x + width, quantized_data, width,
                     color='r',
                     label='Quantized')
    plt.ylabel(yLabel)
    plt.title(title)
    plt.xticks(x + width, xNames)
    plt.legend()
    # save the figure, where?
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--net_file', type=str, nargs='+', help='files created by workflow program')
    args = parser.parse_args()
    plot(args.net_file)
