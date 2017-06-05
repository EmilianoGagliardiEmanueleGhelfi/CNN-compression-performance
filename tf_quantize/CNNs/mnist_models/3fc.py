"""
3 fully connected layer
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf_quantize.CNNs.CNN_utility as cnnu
from tf_quantize.pattern.pattern import ToBeQuantizedNetwork

neurons1 = 1000
neurons2 = 500
neurons3 = 10


class Mnist3Fc(ToBeQuantizedNetwork):
    test_data = None  # initialized in prepare, tuple with input, labels
    input_placeholder_name = 'input'
    label_placeholder_name = 'label'
    output_node_name = 'output'
    net_name = "mnist_net"

    # properties needed to export to pb in workflow. We put checkpoint data, meta graph
    checkpoint_prefix = 'CNNs/mnist_models/net_serializations/3fc/net'
    checkpoint_path = 'CNNs/mnist_models/net_serializations/3fc'
    metagraph_path = 'CNNs/mnist_models/net_serializations/3fc/metagraph.pb'
    output_pb_path = 'CNNs/mnist_models/net_serializations/3fc/output_graph.pb'
    output_quantized_graph = 'CNNs/mnist_models/net_serializations/3fc/quantized_graph.pb'

    def __init__(self):
        self._dataset = None
        self._input_placeholder = None
        self._output_placeholder = None
        self._label_placeholder = None
        self._train_step_node = None
        self._sess = tf.Session()

    def _inference(self):
        """
        Builds the graph as far as is required for running the model forward to make predictions
        :return: input placeholder, output, label placeholder
        """
        x = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_placeholder_name)
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name=self.label_placeholder_name)

        # first layer variables
        W_fc1 = cnnu.weight_variable([784, neurons1])
        b_fc1 = cnnu.bias_variable([neurons1])

        # second layer variables
        W_fc2 = cnnu.weight_variable([neurons1, neurons2])
        b_fc2 = cnnu.bias_variable([neurons2])

        # second layer variables
        W_fc3 = cnnu.weight_variable([neurons2, neurons3])
        b_fc3 = cnnu.bias_variable([neurons3])

        # put the model in the nodes

        layer1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W_fc2) + b_fc2)
        y = tf.add(tf.matmul(layer2, W_fc3), b_fc3, name=self.output_node_name)

        return x, y_, y

    def _loss(self, labels, model_input):
        """
        Adds to the inference graph the ops required to generate loss
        :param labels: the placeholder in the graph for the labels
        :param model_input: the placeholder in the graph for the input
        :return: the loss function node
        """
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model_input))
        return cross_entropy

    def _train(self, loss_node):
        """
        Adds to the graph the ops required to compute and apply gradients
        :param loss_node: the loss function node
        :return: the gradient descent node
        """
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_node)
        return train_step


    """
    from here there is the implementation of the prepare, train pattern
    """

    def prepare(self):
        """
        operation that obtains data and create the computation graph
        """
        self._dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        # assign the test dataset that will be used by the workflow to test this and the quantized net
        self.test_data = (self._dataset.test.images, self._dataset.test.labels)
        self._input_placeholder, self._label_placeholder, self._output_placeholder = self._inference()
        loss_node = self._loss(self._label_placeholder, self._output_placeholder)
        self._train_step_node = self._train(loss_node)

    def train(self):
        """
        train the network
        export checkpoints and the metagraph description
        """
        iterations = 1000
        # initialize the variables
        self._sess.run(tf.global_variables_initializer())
        # training iterations
        for i in range(iterations + 1):
            batch = self._dataset.train.next_batch(100)
            self._sess.run(fetches=self._train_step_node,
                           feed_dict={self._input_placeholder: batch[0], self._label_placeholder: batch[1]})
        self._save()

    def _save(self):
        saver = tf.train.Saver()
        # export checkpoint variables
        saver.save(self._sess, self.checkpoint_prefix, meta_graph_suffix='pb')
        # export the metagraph, first need to obtain the file name of the meta graph from the total path defined as
        # property
        metagraph_filename = self.metagraph_path.split('/')[len(self.metagraph_path.split('/')) - 1]
        tf.train.write_graph(self._sess.graph.as_graph_def(), self.checkpoint_path, metagraph_filename)