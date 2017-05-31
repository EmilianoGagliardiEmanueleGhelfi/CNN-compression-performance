import argparse
import numpy
import configparser
from main import main
from main import writeNetsToFile
from main import train
import os
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description='iterate the compression for various error margin values')
arg_parser.add_argument('ini_file', nargs=1, type=str)
arg_parser.add_argument('initial', nargs=1, type=float)
arg_parser.add_argument('final', nargs=1, type=float)
arg_parser.add_argument('step', nargs=1, type=float)
args = arg_parser.parse_args()

# check the parameters are positive
if args.initial[0] < 0 or args.final[0] < 0 or args.step[0] < 0:
    print "parameter must be positive"
    quit()

# read the config file parameter to generate more new config file with different 
# margin error (and set train to false)
config = configparser.ConfigParser()

# check if the net needs to be trained
config.read(args.ini_file[0])
solver_path = config['CAFFE']['solver_path']
if config['CAFFE'].getboolean('train'):
    train(solver_path)
# maps containig as keys the compression mode and as values lists of 
# nets perfomance information returned by main() for each error margin
netmap = {'dynamic_fixed_point': [], 'minifloat': [], 'integer_power_of_2_weights': []}
for em in numpy.arange(args.initial[0], args.final[0], args.step[0]):
    config.read(args.ini_file[0])
    config.set('CAFFE', 'train', 'False')
    config.set('RISTRETTO', 'error_margin', str(em))
    new_ini = args.ini_file[0].split('.')[0] + '_err_margin_' + str(em).replace('.', '') + '.ini'
    with open(new_ini, 'w') as config_file:
        config.write(config_file)
    list = main(new_ini)
    os.remove(new_ini)
    # at each iteration the original net is the same
    original_net = list[0]
    for net in list[1:]:
        netmap[net.compression_mode].append(net)

# write all the nets in a file
out_file = config['TEST']['benchmark_output_file']
all_nets = [original_net]
for comp_mode in netmap.keys():
    all_nets.extend(netmap[comp_mode])
writeNetsToFile(all_nets, out_file)

# plot the obtained results
# in this map the first array is the x, while the second is the y in the plot
perf_map = {'dynamic_fixed_point': [[], [], []], 'minifloat': [[], [], []], 'integer_power_of_2_weights': [[], [], []]}
for netList in netmap.itervalues():
    for net in netList:
        perf_map[net.compression_mode][0].append(net.error_margin)
        perf_map[net.compression_mode][1].append(net.caffemodel_size)
        perf_map[net.compression_mode][2].append(net.accuracy)

for key in perf_map.keys():
    plt.figure(1)
    plt.subplot(211)
    plt.plot(perf_map[key][0], perf_map[key][1], label=key)
    plt.xlabel("Error Margin")
    plt.ylabel("Model Size")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.subplot(212)
    plt.plot(perf_map[key][0], perf_map[key][2], label=key)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("Error Margin")
    plt.ylabel("Accuracy")

plt.subplot(211)
plt.plot([args.initial[0], arg.final[0]], [original_net.caffemodel_size, original_net.caffemodel_size],
         label="Original")
plt.subplot(212)
plt.plot([args.initial[0], arg.final[0]], [original_net.accuracy, original_net.accuracy], label="Original")

plt.savefig(out_file.split('.')[0] + ".png")
plt.show()
