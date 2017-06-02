"""
This script takes as input a script that defines a cnn, following the prepare, train, evaluate pattern.
"""

import mnist_models.mnist_pattern_implementation as e
import subprocess,signal,platform,os
from net_perf.NetPerformance import NetPerformance
from datetime import datetime
import tensorflow as tf

# performance events
perf_ev_list = ['cache-misses','cache-references','L1-dcache-load-misses','L1-dcache-loads','L1-dcache-stores','LLC-load-misses','LLC-loads','LLC-store-misses','LLC-loads']

"""
quantize takes the model of the net. output graph name is the output of the method freeze. 
output name is the name of the output node of the net
quantized graph name is the output of the method quantize.
requires an env variable pointing to the tensorflow home
"""
def quantize(model):
	quantize_command = os.environ['TF_HOME']+'/bazel-bin/tensorflow/tools/quantization/quantize_graph \
  --input='+model.output_graph_name+' \
  --output_node_names="'+model.output_name+'" --output='+model.quantized_graph_name+' \
  --mode=eightbit'
  	print quantize_command
  	process = subprocess.Popen(quantize_command, shell=True, stdout = subprocess.PIPE)
	process.wait()

"""
This function should restore the graph and test the quantized model
"""
def test_quantized(graph_name,test_data,test_label,accuracy_node_name):
	pass

"""
Runs the function passed as argument inside perf tool. perf writes inside the stderr
-x command is to format output is csv style
-e command is for event stats
-p command is to attach perf to the process with specified pid 
:param function_to_call is a function to call, function are first order obj in python
returns the performance of the net
"""
def cache_perf_test(function_to_call,*args):
	print 'Currently on '+platform.system()
	if platform.system() == 'Linux':
		print 'Testing the model'
		ev_list = ','.join(perf_ev_list)
		perf_command=  'perf stat -x, -e '+ev_list+' -p'+str(os.getpid())
		# with shell = true the shell is the process and the commands are the child, if shell false the command must be splitted and it's the process
		process = subprocess.Popen(perf_command.split(" "),shell=False,stdout=subprocess.PIPE,stderr = subprocess.PIPE)
		acc = function_to_call(*args)
		process.send_signal(signal.SIGINT)
		print 'Terminated, acc ='+str(acc)
		out,err = process.communicate()
		print err
		return NetPerformance(None,acc,err)
	else:
		print 'Perf in not available, testing without cache performance'
	acc = function_to_call(*args)
	return NetPerformance(None,acc,"")

"""
returns the duration in sec of the function call in seconds
"""
def get_test_time(function_to_call, *args):
	init_time = datetime.now()
	function_to_call(*args)
	final_time = datetime.now()
	duration = final_time-init_time
	return duration.total_seconds()

"""
:param function_to_call is a function that takes as input *args, is the function that evaluates the model and gets benchmarked
:param net name is the name of the net
:param test_data is the array representing the test images, used for the size
"""
def get_model_perf(function_to_call,net_name,test_data, *args):
	# get cache performance
	net_perf = cache_perf_test(function_to_call,*args)
	net_perf.name = net_name
    # get test time
    # get test size, test_data is not a tensor, convert it with convert_to_tensor and get its shapes
    # shape[0] is the number of items, shape[1] is product of number of pixel I think
	test_data_size = tf.convert_to_tensor(test_data).get_shape()[0].value
    # convert time returned by get test time into time for each item, in this way we get the average time
	net_perf.test_time = get_test_time(function_to_call,*args)/test_data_size
	print net_perf

def main():
    model = e.MnistNetwork()
    model.prepare()
    model.train()
    print model.evaluate()

    print "Now I'll try to export everything, wish me luck!"
    model.export_to_pb()
    quantize(model)
    get_model_perf(model.evaluate,model.net_name,model.test_data)
    get_model_perf(test_quantized,"model_quant",model.test_data,"blabla",model.test_data,model.test_labels,model.accuracy_node_name)

if __name__ == '__main__':
    main()
