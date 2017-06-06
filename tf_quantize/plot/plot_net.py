import argparse
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import json


from tf_quantize.net_perf.net_performance import NetPerformance

help = 'This script takes in input a pb file, restores the weights of the network, and then computes the ranges for ' \
       'each layer '


def plot(net_file):
    print net_file
    # deserialize file
    with open(net_file, 'r') as data_file:
        nets = json.load(data_file)
    net1 = NetPerformance(json_dict=nets[0])
    net2 = NetPerformance(json_dict=nets[1])
    bar_chart([net1.accuracy/net2.accuracy])


def bar_chart(y):
    """
    Plot a barchart
    :param y: the y-values to plot
    """
    # length of the array, to get the correct indexes
    N = len(y)
    # index of elements
    x = range(N)
    # width of the bar
    width = 0.5
    plt.bar(x, y, width, color="blue")
    # x min, x max, y min, y max
    plt.axis([0.0, 2, 0, 1.5])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # save the figure, where?
    plt.savefig('acc_bar.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--net_file', type=str, nargs=1, help='file created by workflow program')
    args = parser.parse_args()
    plot(args.net_file[0])

