"""
This script takes as input a script that defines a cnn, following the prepare, train, evaluate pattern.
"""

import mnist_models.mnist_pattern_implementation as e
import os
import subprocess

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



def main():
    model = e.MnistNetwork()
    model.prepare()
    model.train()
    print model.evaluate()

    print "Now I'll try to export everything, wish me luck!"
    model.export_to_pb()
    quantize(model)


if __name__ == '__main__':
    main()
