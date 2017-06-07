import argparse
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import json


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
        net_list.append([net1, net2])
    bar_chart([x[1].accuracy/x[0].accuracy for x in net_list])


def bar_chart(y):
    """
    Plot a barchart
    :param y: the y-values to plot
    """
    # length of the array, to get the correct indexes
    N = len(y)
    # index of elements
    x = range(1,N+1)
    # width of the bar
    width = 0.2
    plt.bar(x, y, width, color="blue")
    # x min, x max, y min, y max
    plt.axis([0.0, 2*N, 0, 1.5])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # save the figure, where?
    plt.savefig('acc_bar.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--net_file', type=str, nargs='+', help='files created by workflow program')
    args = parser.parse_args()
    plot(args.net_file)

