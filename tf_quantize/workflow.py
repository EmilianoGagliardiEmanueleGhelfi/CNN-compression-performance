"""
This script takes as input a script that defines a cnn, following the prepare, train, evaluate pattern.
"""

import mnist_models.mnist_pattern_implementation as e
import subprocess, signal, platform, os
from net_perf.NetPerformance import NetPerformance
from datetime import datetime
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

# performance events
perf_ev_list = ['cache-misses', 'cache-references', 'L1-dcache-load-misses', 'L1-dcache-loads', 'L1-dcache-stores',
                'LLC-load-misses', 'LLC-loads', 'LLC-store-misses', 'LLC-loads']


def quantize(model):
    """
    Quantize the graph
    quantize takes the model of the net. output graph name is the output of the method freeze.
    output name is the name of the output node of the net
    quantized graph name is the output of the method quantize.
    requires an env variable pointing to the tensorflow home
    :param model: the net implementing the abstract class
    """
    quantize_command = os.environ['TF_HOME'] + '/bazel-bin/tensorflow/tools/quantization/quantize_graph \
  --input=' + model.output_pb_path + ' \
  --output_node_names="' + model.output_name + '" --output=' + model.output_quantized_graph + ' \
  --mode=eightbit'
    process = subprocess.Popen(quantize_command, shell=True, stdout=subprocess.PIPE)
    process.wait()


def export_to_pb(model):
    """
    This method export to pb the network defined in the model param. The train have to be already done and the
    variables and the meta graph have to be already exported. The method assumes that the path of the meta graph file
    is model.metagraph_path and the checkpoint have been stored with the model.checkpoint_prefix. The output file will
    be stored with model.output_pb_path
    :param model: an instance of a ToBeQuantizedNetwork subclass
    """
    input_graph_name = model.metagraph_path
    output_graph_name = model.output_pb_path

    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = model.checkpoint_prefix
    output_node_names = model.output_node_name
    restore_op_name = tf.train.latest_checkpoint(model.checkpoint_prefix)
    filename_tensor_name = "save/Const:0"
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_name, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_name, clear_devices, "")


def cache_perf_test(function_to_call, *args):
    """
    Runs the function passed as argument inside perf tool. perf writes inside the stderr
    -x command is to format output is csv style
    -e command is for event stats
    -p command is to attach perf to the process with specified pid
    :param function_to_call: is a function to call, function are first order obj in python
    :param args: arguments required from function to call
    :return: the performance of the net
    """
    print 'Running on ' + platform.system()
    if platform.system() == 'Linux':
        print 'Testing the model'
        ev_list = ','.join(perf_ev_list)
        perf_command = 'perf stat -x, -e ' + ev_list + ' -p' + str(os.getpid())
        # with shell = true the shell is the process and the commands are the child
        # if shell false the command must be splitted and it's the process
        process = subprocess.Popen(perf_command.split(" "), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        acc = function_to_call(*args)
        process.send_signal(signal.SIGINT)
        print 'Terminated, accuracy =' + str(acc)
        out, err = process.communicate()
        print err
        return NetPerformance(None, acc, err)
    else:
        print 'Perf in not available, testing without cache performance'
    acc = function_to_call(*args)
    return NetPerformance(None, acc, "")


def get_test_time(function_to_call, *args):
    """
    Get the duration of the function_to_call parameter
    :param function_to_call: function to test
    :param args: arguments needed from the function
    :return: The duration in seconds of the function to call
    """
    init_time = datetime.now()
    function_to_call(*args)
    final_time = datetime.now()
    duration = final_time - init_time
    return duration.total_seconds()


def get_model_perf(function_to_call, net_name, test_data, *args):
    """
    Test the performance of the function to call in terms of cache and time
    :param function_to_call: is a function that takes as input *args, is the function that evaluates the model and gets benchmarked
    :param net_name: the name of the net
    :param test_data: The tensor containing the test data, in order to get the avg forward time
    :param args:  arguments needed from the function to call
    :return: a NetPerformance object containing the performance of the net
    """
    # get cache performance
    net_perf = cache_perf_test(function_to_call, *args)
    net_perf.name = net_name
    # get test time
    # get test size, test_data is not a tensor, convert it with convert_to_tensor and get its shapes
    # shape[0] is the number of items, shape[1] is product of number of pixel I think
    test_data_size = tf.convert_to_tensor(test_data).get_shape()[0].value
    # convert time returned by get test time into time for each item, in this way we get the average time
    net_perf.test_time = get_test_time(function_to_call, *args) / test_data_size
    print net_perf


def evaluate(output_node, test_data, labels, input_placeholder_node, label_placeholder_node):
    """
    Takes a graph as input, attaches an accuracy node and evaluates it
    :param output_node output node of the net, used for accuracy
    :param labels is the tensor containing labels of test data
    :param input_placeholder_node node used for placeholder for input
    :param label_placeholder_node node used as placeholder for labels
    :param test_data test images
    :return: the accuracy of the net
    """
    sess = tf.Session()
    correct_prediction = tf.equal(tf.argmax(output_node, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = sess.run(accuracy,
                             feed_dict={input_placeholder_node: test_data,
                                        label_placeholder_node: labels})
    return accuracy


def restore(meta_graph_path, model):
    """
    Takes a pb path of a freezed graph and restores it. :0 after the name is to get the first of the list (?)
    :param model is an instance of the abstract class
    :param meta_graph_path is the path of the freezed graph
    :return: a triple (ouput_node,input_placeholder,label_placeholder
    """
    # take the graph
    saver = tf.train.import_meta_graph(meta_graph_path)
    graph = tf.get_default_graph()
    # access placeholder
    input_placeholder = graph.get_tensor_by_name(model.input_placeholder_name+":0")
    label_placeholder = graph.get_tensor_by_name(model.label_placeholder_name+":0")
    output_node = graph.get_tensor_by_name(model.output_node_name+":0")
    return output_node, input_placeholder, label_placeholder


def main():
    model = e.MnistNetwork()
    model.prepare()
    model.train()
    print "Exporting the graph"
    # export the trained model to pb
    export_to_pb(model)
    # quantize the trained model
    quantize(model)
    # restore the real model
    output_node, input_placeholder, label_placeholder = restore(model.output_pb_path, model)
    # get performance of the model
    get_model_perf(evaluate, model.net_name, model.test_data.images, output_node, model.test_data[0],
                   model.test_data[1], input_placeholder, label_placeholder)
    # the same with the quantized model
    output_node, input_placeholder, label_placeholder = restore(model.output_quantized_graph, model)
    get_model_perf(evaluate, "model_quant", model.test_data.images, output_node, model.test_data[0],
                   model.test_data[1], input_placeholder, label_placeholder)


if __name__ == '__main__':
    main()
