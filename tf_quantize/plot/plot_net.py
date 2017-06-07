import argparse
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import json
import numpy as np

from tf_quantize.net_perf.net_performance import NetPerformance

help = 'This script takes in input a pb file, restores the weights of the network, and then computes the ranges for ' \
       'each layer '


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

    bar_chart(net_list)


def bar_chart(nets):
    """
    Plot a barchart
    :param y: the y-values to plot as couple
    """
    # plot accuracy
    accuracy_list = [n.accuracy for n in nets]
    original_acc = [accuracy_list[i] for i in range(0, len(accuracy_list), 2)]
    quantized_acc = [accuracy_list[i] for i in range(1, len(accuracy_list), 2)]
    # length of the array, to get the correct indexes
    N = len(original_acc)
    # index of elements
    x = np.arange(N)
    # width of the bar
    width = 0.2
    rects1 = plt.bar(x, original_acc, width,
                     color='b',
                     label='Original')
    rects2 = plt.bar(x + width, quantized_acc, width,
                     color='r',
                     label='Quantized')
    plt.xlabel('Nets')
    plt.ylabel('Accuracy')
    plt.title('Comparison of accuracies')
    plt.xticks(x + width, (nets[i].name for i in range(0, len(nets), 2)))
    plt.legend()
    # x min, x max, y min, y max
    # plt.axis([0.0, 2*N, 0, 1.5])
    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    # save the figure, where?
    plt.savefig('acc_bar.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--net_file', type=str, nargs='+', help='files created by workflow program')
    args = parser.parse_args()
    plot(args.net_file)
