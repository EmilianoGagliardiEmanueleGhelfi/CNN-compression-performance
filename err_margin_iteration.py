import argparse
import numpy
import configparser
from main import main
from main import writeNetsToFile
import os
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description = 'iterate the compression for various error margin values')
arg_parser.add_argument('ini_file', nargs=1, type = str)
arg_parser.add_argument('initial', nargs=1, type = float)
arg_parser.add_argument('final', nargs=1, type = float)
arg_parser.add_argument('step', nargs=1, type = float)
args = arg_parser.parse_args()

# check the parameters are positive
if args.initial[0] < 0 or args.final[0] < 0 or args.step[0] < 0:
	print "parameter must be positive"
	quit()

# read the config file parameter to generate more new config file with different 
# margin error (and set train to false)
config = configparser.ConfigParser()
# maps containig as keys the compression mode and as values lists of 
# nets perfomance information returned by main() for each error margin
netmap = {'dynamic_fixed_point': [], 'minifloat': [], 'integer_power_of_2_weights': []}
for em in numpy.arange(args.initial[0], args.final[0], args.step[0]):
	config.read(args.ini_file[0])
	config.set('CAFFE', 'train', 'False')
	config.set('RISTRETTO', 'error_margin', str(em))
	new_ini = args.ini_file[0].split('.')[0] + '_err_margin_' + str(em).replace('.', '') + '.ini'
	with open (new_ini, 'w') as config_file:
		config.write(config_file)
	list = main(new_ini)
	os.remove(new_ini)
	# at each iteration the original net is the same
	original_net = list[0] 
	for net in list[1 :]:
		netmap[net.compression_mode].append(net)

# write all the nets in a file
config.read(args.ini_file[0])
out_file = config['TEST']['benchmark_output_file']
all_nets = [original_net]
for comp_mode in netmap.keys():
	all_nets.append(netmap[compression_mode])

# plot the obtained results
