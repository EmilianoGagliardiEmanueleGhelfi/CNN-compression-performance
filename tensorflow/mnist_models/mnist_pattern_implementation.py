import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import freeze_graph


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


class MnistNetwork:
    def __init__(self):
        self.dataset = None
        self.input = None
        self.output = None
        self.label = None
        self.train_step = None
        self.test_step = None
        self.sess = tf.Session()
        self.input_name = 'input'
        self.output_name = 'output'
        self.label_name = 'label'
        self.checkpoint_path = 'mnist_models/models'
        self.net_name = 'mnist_network'
        self.checkpoint_prefix = self.checkpoint_path + '/' + self.net_name
        self.output_graph_name = 'mnist_models/models/output_graph.pb'
        self.quantized_graph_name = 'mnist_models/models/quantized_graph.pb'
        self.accuracy_node_name = 'accuracy'

    """
    The model and the training method is defined as implementation of the inference, loss, training pattern
    """

    def _inference(self):
        """
        Builds the graph as far as is required for running the model forward to make predictions
        :return: input placeholder, output, label placeholder
        """
        x = tf.placeholder(tf.float32, shape=[None, 784], name=self.input_name)
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name=self.label_name)

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

        y_conv = tf.add(tf.matmul(h_fc1, W_fc2),b_fc2,name=self.output_name)

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
        # gradient descent with step 0.5
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_node)
        return train_step

    def _test(self, labels, output):
        """
        Adds to the graph the ops required to compute the accuracy
        :param labels: the placeholder in the graph for the labels
        :param output: the output node of the model
        :return: the accuracy node
        """
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name=self.accuracy_node_name)
        return accuracy

    """
    from here there is the implementation of the prepare, train, evaluate pattern
    """

    def prepare(self):
        """
        operation that obtains data and create the computation graph
        """
        self.dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.input, self.output, self.label = self._inference()
        loss_node = self._loss(self.label, self.output)
        self.train_step = self._train(loss_node)
        self.test_step = self._test(self.label, self.output)

    def train(self):
        """
        train the network 
        export checkpoints and the metagraph description 
        
        :return: None 
        """
        saver = tf.train.Saver()
        iterations = 1
        # initialize the variables
        self.sess.run(tf.global_variables_initializer())
        # training iterations
        for i in range(iterations + 1):
            batch = self.dataset.train.next_batch(100)
            self.sess.run(fetches=self.train_step, feed_dict={self.input: batch[0], self.label: batch[1]})

            # evaluate and print the train accuracy
            if i % 100 == 0:
                train_acc = self.sess.run(fetches=self.test_step, feed_dict={self.input: batch[0], self.label: batch[1]})
                print ('Step %d train accuracy %g' % (i, train_acc))
        # training finished, export model
        saver.save(self.sess, self.checkpoint_prefix, meta_graph_suffix='pb')
        # saver.export_meta_graph(model_name, as_text=True)
        tf.train.write_graph(self.sess.graph.as_graph_def(), self.checkpoint_path, self.net_name + '.pb')

    def evaluate(self):
        accuracy = self.sess.run(self.test_step,
                                 feed_dict={self.input: self.dataset.test.images,
                                            self.label: self.dataset.test.labels})
        return accuracy

    def export_to_pb(self):
        input_graph_name = self.checkpoint_prefix + '.pb'
        output_graph_name = self.output_graph_name

        input_saver_def_path = ""
        input_binary = False
        input_checkpoint_path = self.checkpoint_prefix  # maybe need to add iterations number
        output_node_names = self.output_name
        restore_op_name = tf.train.latest_checkpoint(self.checkpoint_prefix)
        filename_tensor_name = "save/Const:0"
        clear_devices = True

        freeze_graph.freeze_graph(input_graph_name, input_saver_def_path,
                                  input_binary, input_checkpoint_path,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, output_graph_name, clear_devices, "")

        
