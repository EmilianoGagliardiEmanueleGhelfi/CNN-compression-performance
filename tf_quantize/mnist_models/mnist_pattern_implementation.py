import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pattern.pattern import ToBeQuantizedNetwork


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class MnistNetwork(ToBeQuantizedNetwork):
    # properties needed to evaluate the quantized network in workflow
    test_iterations = 1
    test_data = None  # initialized in prepare, tuple with input, labels
    input_placeholder_name = 'input'
    label_placeholder_name = 'label'
    output_node_name = 'output'
    net_name ="mnist_net"

    # properties needed to export to pb in workflow. We put checkpoint data, meta graph
    checkpoint_prefix = 'mnist_models/models/net'
    checkpoint_path = 'mnist_models/models'
    metagraph_path = 'mnist_models/models/metagraph.pb'
    output_pb_path = 'mnist_models/models/output_graph.pb'
    output_quantized_graph = 'mnist_models/models/quantized_graph.pb'

    def __init__(self):
        self._dataset = None
        self._input_placeholder = None
        self._output_placeholder = None
        self._label_placeholder = None
        self._train_step_node = None
        self._sess = tf.Session()

    """
    The model and the training method is defined as implementation of the inference, loss, training pattern
    """

    def _inference(self):
        """
        Builds the graph as far as is required for running the model forward to make predictions
        :return: input placeholder, output, label placeholder
        """
        x = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_placeholder_name)
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name=self.label_placeholder_name)

        # create the layers

        # first convolution pooling layer
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # second convolution pooling layer
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # densely connected layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name=self.output_node_name)

        return x, y_conv, y_

    def _loss(self, labels, model_input):
        """
        Adds to the inference graph the ops required to generate loss
        :param labels: the placeholder in the graph for the labels
        :param model_input: the placeholder in the graph for the input
        :return: the loss function node
        """
        # loss function
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
    from here there is the implementation of the prepare, train, evaluate pattern
    """

    def prepare(self):
        """
        operation that obtains data and create the computation graph
        """
        self._dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        # assign the test dataset that will be used by the workflow to test this and the quantized net
        self.test_data = (self._dataset.test.images, self._dataset.test.labels)
        self._input_placeholder, self._output_placeholder, self._label_placeholder = self._inference()
        loss_node = self._loss(self._label_placeholder, self._output_placeholder)
        self._train_step_node = self._train(loss_node)

    def train(self):
        """
        train the network 
        export checkpoints and the metagraph description
        """
        iterations = 1
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
