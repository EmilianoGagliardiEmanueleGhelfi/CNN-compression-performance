"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
import pprint

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import os

import cifar10_processing
import CNNs.CNN_utility as cnnu
from pattern.pattern import ToBeQuantizedNetwork

BATCH_SIZE = 50
STEPS = 200000

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_processing.IMG_SIZE
NUM_CLASSES = cifar10_processing.NUM_CLASSES


class Cifar10Network(ToBeQuantizedNetwork):
    # properties needed to evaluate the quantized network in workflow
    test_iterations = 1
    test_data = []  # initialized in prepare, tuple with input, labels
    input_placeholder_name = 'input'
    label_placeholder_name = 'label'
    output_node_name = 'output'
    net_name = "cifar10_net"

    # properties needed to export to pb in workflow. We put checkpoint data, meta graph
    checkpoint_prefix = 'CNNs/cifar10_models/net_serialization/2conv_2fc/net'
    checkpoint_path = 'CNNs/cifar10_models/net_serialization/2conv_2fc'
    metagraph_path = 'CNNs/cifar10_models/net_serialization/2conv_2fc/metagraph.pb'
    output_pb_path = 'CNNs/cifar10_models/net_serialization/2conv_2fc/output_graph.pb'
    output_quantized_graph = 'CNNs/cifar10_models/net_serialization/2conv_2fc/quantized_graph.pb'

    def __init__(self):
        self._dataset = None
        self.test_data = []
        self._input_placeholder = None
        self._output_placeholder = None
        self._label_placeholder = None
        self._train_step_node = None
        self._sess = tf.Session()
        self._accuracy_node = None
        self._loss_node = None
        self.serialization_path = 'CNNS/cifar10_models/net_serialization'

    def _loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
          logits: Logits from inference().
          labels: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]
        Returns:
          Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def _train(self, total_loss):
        """Train CIFAR-10 model.
        Create an optimizer and apply to all trainable variables.
        Args:
          total_loss: Total loss from loss().
        Returns:
          train_op: op for training.
        """
        train_step = tf.train.AdamOptimizer(0.01).minimize(total_loss)

        return train_step

    def _inference(self):
        """Build the CIFAR-10 model.
        Args:
          images: Images returned from distorted_inputs() or inputs().
        Returns:
          Logits.
        """
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
        #

        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name=self.input_placeholder_name)
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name=self.label_placeholder_name)
        # conv1
        kernel1 = cnnu.weight_variable([5, 5, 3, 64])
        conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = cnnu.bias_variable([64])
        pre_activation = tf.nn.bias_add(conv1, biases1)
        relu1 = tf.nn.relu(pre_activation)

        # pool1
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        kernel2 = cnnu.weight_variable([5, 5, 64, 64])
        conv2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = cnnu.bias_variable([64])
        pre_activation2 = tf.nn.bias_add(conv2, biases2)
        relu2 = tf.nn.relu(pre_activation2)

        # norm2
        norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [-1, 8 * 8 * 64])
        weights_1 = cnnu.weight_variable([8 * 8 * 64, 384])

        biases_1 = cnnu.bias_variable([384])
        local3 = tf.nn.relu(tf.matmul(reshape, weights_1) + biases_1, name='local3')

        # local4
        weights_2 = cnnu.weight_variable([384, 192])
        biases_2 = cnnu.bias_variable([192])
        local4 = tf.nn.relu(tf.matmul(local3, weights_2) + biases_2, name='local4')

        # linear layer(WX + b),
        # We don't apply softmax here because
        # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
        # and performs the softmax internally for efficiency.
        weights_final = cnnu.weight_variable([192, NUM_CLASSES])
        biases_final = cnnu.bias_variable([NUM_CLASSES])
        softmax_linear = tf.add(tf.matmul(local4, weights_final), biases_final, name=self.output_node_name)

        return x, softmax_linear, y_

    """
    from here there is the implementation of the prepare, train, evaluate pattern
    """

    def prepare(self):
        """
        operation that obtains data and create the computation graph
        """
        cifar10_processing.maybe_download_and_extract()
        images, _, labels = cifar10_processing.load_training_data()
        # assign the test dataset that will be used by the workflow to test this and the quantized net
        test_images, _, test_labels = cifar10_processing.load_test_data()
        # create an instance of dataset class
        self._dataset = DataSet(images, labels, one_hot=True, reshape=False)
        self.test_data = (test_images, test_labels)
        self._input_placeholder, self._output_placeholder, self._label_placeholder = self._inference()
        self._loss_node = self._loss(self._output_placeholder, self._label_placeholder)
        self._accuracy_node = self.accuracy(self._output_placeholder, self._label_placeholder)
        self._train_step_node = self._train(self._loss_node)

    def train(self):
        """
        train the network
        export checkpoints and the metagraph description
        """
        # initialize the variables
        saver = tf.train.Saver()
        #if os.path.exists(self.checkpoint_prefix + '.pb'):
            #saver.restore(self._sess, self.checkpoint_prefix)
        #else:
        self._sess.run(tf.global_variables_initializer())
        # training iterations
        for i in range(STEPS + 1):
            batch = self._dataset.next_batch(BATCH_SIZE)
            self._sess.run(fetches=self._train_step_node,
                           feed_dict={self._input_placeholder: batch[0], self._label_placeholder: batch[1]})
            if i % 500 == 0:
                # run the accuracy node
                acc = self._sess.run(fetches=self._accuracy_node,
                                     feed_dict={self._input_placeholder: self.test_data[0],
                                                self._label_placeholder: self.test_data[1]})
                tr_acc = self._sess.run(fetches=self._accuracy_node,
                           feed_dict={self._input_placeholder: batch[0], self._label_placeholder: batch[1]})
                print "Iteration " + str(i) + ", Acc " + str(acc) + " Training acc " + str(tr_acc)
                saver.save(self._sess, self.checkpoint_prefix, meta_graph_suffix='pb')

    def accuracy(self, output_node, label_placeholder):
        """
        Get the output node and attach to it the accuracy node
        :param output_node: the output of the net
        :param label_placeholder:
        :return: the accuracy node
        """
        correct_prediction = tf.equal(tf.argmax(output_node, 1), tf.argmax(label_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _save(self):
        saver = tf.train.Saver()
        # export checkpoint variables
        saver.save(self._sess, self.checkpoint_prefix, meta_graph_suffix='pb')
        # export the metagraph, first need to obtain the file name of the meta graph from the total path defined as
        # property
        metagraph_filename = self.metagraph_path.split('/')[len(self.metagraph_path.split('/')) - 1]
        tf.train.write_graph(self._sess.graph.as_graph_def(), self.checkpoint_path, metagraph_filename)
