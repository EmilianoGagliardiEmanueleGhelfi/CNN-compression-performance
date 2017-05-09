import caffe
import argparse
import configparser
import sys
import os
from caffe.proto import caffe_pb2
from google.protobuf.text_format import Merge
import subprocess 


# train the network, you will find the weights files as in the definition of the
# snapshot_prefix in the solver .prototxt file
def train(solver_name):
	solver = caffe.SGDSolver(solver_name)
	caffe.set_mode_cpu()
	solver.solve()

#
# compress the network using the passed method, the output files will be in the
# net directory
#
def compress(solver_name, mode, error_margin=1, iterations=1000):
	# obtain the oputput file of the training (caffe_model)
	solver = caffe_pb2.SolverParameter()
	f = open(solver_name, 'r')
	Merge(f.read(), solver)
	# obtain the net file path
	net = solver.net
	# obtain the weigths file path
	snapshot_prefix = solver.snapshot_prefix
	max_iter = solver.max_iter
	weigths = snapshot_prefix + '_iter_' + str(max_iter) + '.caffemodel'
	# path output file of the compression
	output = net.split('.')[0] + '_' + mode + '.prototxt'
	# build the ristretto command
	command = os.environ['RISTRETTOPATH'] + "/build/tools/ristretto quantize" + "\
		--model=" + net + "\
		--weights=" + weigths + "\
		--model_quantized=" + output + "\
		--iterations=" + iterations + "\
		--trimming_mode=" + mode + "\
		--error_margin=" + error_margin
	process = subprocess.Popen(command, shell=True, stdout = subprocess.PIPE)
	process.wait()



if __name__ == "__main__":
	# get the config file as parameter
	arg_parser = argparse.ArgumentParser(description='TODO')#TODO
	arg_parser.add_argument('ini_file', nargs=1, type = str)
	args = arg_parser.parse_args()
	# parse the config file
	config = configparser.ConfigParser()
	config.read(args.ini_file[0])
	# caffe parameters
	to_train = config['CAFFE'].getboolean('train')
	solver_path = config['CAFFE']['solver_path']
	# ristretto parameters
	error_margin = config['RISTRETTO']['error_margin']
	iterations = config['RISTRETTO']['iterations']

	# go in the directory of the solver
	os.chdir(os.path.dirname(solver_path))
	solver_name = solver_path.split('/')[len(solver_path.split('/'))-1]

	if to_train:
		train(solver_name)

	for compression_mode in ('dynamic_fixed_point', 'minifloat', 'integer_power_of_2_weights'):
		if config['RISTRETTO'][compression_mode]:
			compress(solver_name, compression_mode, error_margin, iterations)