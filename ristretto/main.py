import caffe
import argparse
import configparser
import sys
import os
import subprocess

from utility.solver_reader import SolverReader
from utility.net_performance import NetPerformance
from utility.filter_utility import vis_square,weights_hist

#
# train the network, you will find the weights files as in the definition of the
# snapshot_prefix in the solver .prototxt file
#
def train(solver_filename):
	solver = caffe.SGDSolver(str(solver_filename))
	caffe.set_mode_cpu()
	solver.solve()

#
# compress the network using the passed method, the output files will be in the
# net directory
#
def compress(solver_filename, mode, error_margin=1, iterations=1000):
	solver_reader = SolverReader(solver_filename)
	net = solver_reader.solver.net
	max_iter = solver_reader.solver.max_iter
	weights = solver_reader.weightsFilename()
	output = solver_reader.compressionOutputFilename(mode, error_margin)
	# build the ristretto command
	command = os.environ['RISTRETTOPATH'] + "/build/tools/ristretto quantize" + "\
		--model=" + net + "\
		--weights=" + weights + "\
		--model_quantized=" + output + "\
		--iterations=" + iterations + "\
		--trimming_mode=" + mode + "\
		--error_margin=" + error_margin	
	process = subprocess.Popen(command, shell=True, stdout = subprocess.PIPE)
	process.wait()

#
# create the new solver file and call train
#
def fine_tune(solver_filename, compression_mode, error_margin):
	# create a new solver file, equal to the input
	# but with compressed network as target
	solver_reader = SolverReader(solver_filename)
	fine_tune_solver_file = solver_reader.createFineTuneSolverFile(compression_mode, error_margin)
	train(fine_tune_solver_file)

#
# obtain the name of the net and the name of the weights file
# from the solver file and call valgrind 
# need_to_test_time if need to call caffe time
#
def test(solver_filename, test_iterations,performance_path,need_to_test_time,compression_mode = None):
	solver_reader = SolverReader(solver_filename)
	net = solver_reader.solver.net
	weights = solver_reader.weightsFilename()
	# create the folder if not exists
	if not os.path.exists(performance_path):
		os.mkdir(performance_path)
	# create the cachegrind output file
	net_name = net.split('/')[len(net.split('/'))-1].split('.')[0]
	cachegrind_out_file = performance_path + '/' + net_name + '_cachegrind.txt'
	# call the command for testing
	
	command = "valgrind --tool=cachegrind \
		--cachegrind-out-file=" + cachegrind_out_file + "\
		$RISTRETTOPATH/build/tools/caffe test \
		-model=" + net + "\
		-weights=" + weights + "\
		-iterations=" + str(test_iterations)	
	output = subprocess.check_output(command, shell=True,stderr=subprocess.STDOUT)
	test_time_val = None
	if need_to_test_time:
		test_time_val=evaluate_test_time(solver_filename,test_iterations)
	return NetPerformance(net,weights,cachegrind_out_file,output,compression_mode,test_time_val)
#
# Test the inference time of the net and returns it
#
def evaluate_test_time(solver_filename,test_iterations):
	solver_reader = SolverReader(solver_filename)
	net = solver_reader.solver.net
	weights = solver_reader.weightsFilename()
	command = "$RISTRETTOPATH/build/tools/caffe time -model "+net+" \
				-weights "+weights+ " -iterations " + str(test_iterations)
	output = subprocess.check_output(command,shell=True,stderr=subprocess.STDOUT)
	print output
	# parse the output in order to get test time
	lines = output.split("\n")
	line = [x for x in lines if "Average Forward pass" in x][0]
	test_time = line.split(" ")[len(line.split(" "))-2]
	return float(test_time)
#
# writes the net list to the output file
#
def writeNetsToFile(net_list,outputfile):
	output_folder = os.path.dirname(outputfile)
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	f = open(outputfile,'w')
	for net in net_list:
		f.write('[NET NAME: '+net.net+']\n\n')
		f.write(str(net))
		f.write("\n\n")
	f.close()

#
# Visualize the weights for the first convolutional layer
#
def vis_weights(solver_filename):
	solver_reader = SolverReader(solver_filename)
	model_def = solver_reader.solver.net
	model_weights = solver_reader.weightsFilename()
	net = caffe.Net(str(model_def),      # defines the structure of the model
                str(model_weights),  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
	# the parameters are a list of [weights, biases]
	filters = net.params['conv1'][0].data
	#print filters.shape
	vis_square(filters.transpose(0, 2, 3,1))
	weights=[]
	for layer_name,param in net.params.iteritems():
		weights.extend(net.params[layer_name][0].data.flatten())
	weights_hist(weights)


def main(inifile):
	# parse the config file
	config = configparser.ConfigParser()
	config.read(inifile)
	to_train = config['CAFFE'].getboolean('train')
	solver_path = config['CAFFE']['solver_path']
	error_margin = config['RISTRETTO']['error_margin']
	iterations = config['RISTRETTO']['iterations']
	test_iterations = config['TEST']['iterations']
	cache_performance = config['TEST'].getboolean('cache_performance')
	performance_path = config['TEST']['cachegrind_output_folder']
	test_time = config['TEST'].getboolean('test_time')
	# list of net performances in order to write into file
	net_list = []

	# train the input net
	if to_train:
		train(solver_path)

	# compress the net with the modes in the ini file and fine tune
	for compression_mode in ('dynamic_fixed_point', 'minifloat', 'integer_power_of_2_weights'):
		if config['RISTRETTO'].getboolean(compression_mode):
			compress(solver_path, compression_mode, error_margin, iterations)
			fine_tune(solver_path, compression_mode, error_margin)

	# run the test through valgrind for the non compressed network
	if cache_performance:
		net_list.append(test(solver_path, test_iterations,performance_path,test_time))
		solver_reader = SolverReader(solver_path)
		for compression_mode in ('dynamic_fixed_point', 'minifloat', 'integer_power_of_2_weights'):
			if config['RISTRETTO'].getboolean(compression_mode):
				fine_tune_solver = solver_reader.fineTuneSolverName(compression_mode, error_margin)
				net_tested = test(fine_tune_solver, test_iterations, performance_path, test_time, compression_mode)
				net_tested.error_margin = error_margin
				net_list.append(net_tested)
	return net_list

if __name__=='__main__':
	# get the config file as parameter
	arg_parser = argparse.ArgumentParser(description='TODO')#TODO
	arg_parser.add_argument('ini_file', nargs=1, type = str)
	args = arg_parser.parse_args()
	net_list = main(args.ini_file[0])

	# write all nets to file
	config = configparser.ConfigParser()
	config.read(inifile)
	benchmark_output_file = config['TEST']['benchmark_output_file']
	writeNetsToFile(net_list, benchmark_output_file)