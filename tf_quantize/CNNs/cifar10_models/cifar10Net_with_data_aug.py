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
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes

import cifar10_processing
import CNNs.CNN_utility as cnnu
from pattern.pattern import ToBeQuantizedNetwork
import logging
from matplotlib import pyplot as plt

BATCH_SIZE = 64
STEPS = 10000

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_processing.IMG_SIZE
NUM_CLASSES = cifar10_processing.NUM_CLASSES

img_size_cropped = 24

# setup logging
logging.basicConfig(filename='CNNs/cifar10_models/net_serialization/2conv_2fc_da/accuracy.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s')


class Cifar10NetworkWithDataAug(ToBeQuantizedNetwork):
    # properties needed to evaluate the quantized network in workflow
    test_iterations = 1
    test_data = []  # initialized in prepare, tuple with input, labels
    input_placeholder_name = 'input'
    label_placeholder_name = 'label'
    output_node_name = 'network_1/output'
    net_name = "cifar10_net"

    # properties needed to export to pb in workflow. We put checkpoint data, meta graph
    checkpoint_prefix = 'CNNs/cifar10_models/net_serialization/2conv_2fc_da/net'
    checkpoint_path = 'CNNs/cifar10_models/net_serialization/2conv_2fc_da'
    metagraph_path = 'CNNs/cifar10_models/net_serialization/2conv_2fc_da/metagraph.pb'
    output_pb_path = 'CNNs/cifar10_models/net_serialization/2conv_2fc_da/output_graph.pb'
    output_quantized_graph = 'CNNs/cifar10_models/net_serialization/2conv_2fc_da/quantized_graph.pb'

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
        self.global_step = None
        self._output_training_node = None
        self._global_step = None
        self._train_input_placeholder = None

    def plot_images(self, images, cls_true, cls_pred=None, smooth=True, class_names=None):

        assert len(images) == len(cls_true) == 9

        # Create figure with sub-plots.
        fig, axes = plt.subplots(3, 3)

        # Adjust vertical spacing if we need to print ensemble and best-net.
        if cls_pred is None:
            hspace = 0.3
        else:
            hspace = 0.6
        fig.subplots_adjust(hspace=hspace, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'

            # Plot image.
            ax.imshow(images[i, :, :, :],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()


    def pre_process_image(self, image, training):
        # This function takes a single image as input,
        # and a boolean whether to build the training or testing graph.
        # Used for data augmentation

        if training:
            # For training, add the following to the TensorFlow graph.
            # Randomly crop the input image.
            image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, 3])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Randomly adjust hue, contrast and saturation.
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # For training, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=img_size_cropped,
                                                           target_width=img_size_cropped)

        return image

    def pre_process(self, images, training):
        # Use TensorFlow to loop over all the input images and call
        # the function above which takes a single image as input.
        if training:
            images.set_shape([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
        images = tf.map_fn(lambda image: self.pre_process_image(image, training), images)

        return images

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

    def _train(self, total_loss, global_step=None):
        """Train CIFAR-10 model.
        Create an optimizer and apply to all trainable variables.
        Args:
          total_loss: Total loss from loss().
        Returns:
          train_op: op for training.
        """
        train_step = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)

        return train_step

    def _inference(self, training=True):
        """Build the CIFAR-10 model.
        Returns:
          Logits.
        """
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
        #
        if not training:
            x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name=self.input_placeholder_name)
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name=self.label_placeholder_name)
        else:
            x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name=self.input_placeholder_name+'_train')
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name=self.label_placeholder_name)

        with tf.variable_scope('network', reuse=not training):
            img = x
            # do preprocessing if needed
            img = self.pre_process(images=img, training=training)

            # conv1
            kernel1 = cnnu.weight_variable([5, 5, 3, 64], name='kernel1')
            conv1 = tf.nn.conv2d(img, kernel1, [1, 1, 1, 1], padding='SAME')
            biases1 = cnnu.bias_variable([64], name='bias1')
            pre_activation = tf.nn.bias_add(conv1, biases1)
            relu1 = tf.nn.relu(pre_activation)

            # pool1
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            # norm1
            # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            #                 name='norm1')

            # conv2
            kernel2 = cnnu.weight_variable([5, 5, 64, 64], name='kernel2')
            conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
            biases2 = cnnu.bias_variable([64], name='bias2')
            pre_activation2 = tf.nn.bias_add(conv2, biases2)
            relu2 = tf.nn.relu(pre_activation2)

            # norm2
            # norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            #                  name='norm2')
            # pool2
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            # local3
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [-1, 6 * 6 * 64])
            weights_1 = cnnu.weight_variable([6 * 6 * 64, 256], name='local_weights_3')
            biases_1 = cnnu.bias_variable([256], name='local_bias_3')
            local3 = tf.nn.relu(tf.matmul(reshape, weights_1) + biases_1, name='local3')

            # local4
            weights_2 = cnnu.weight_variable([256, 192], name='local_weights_4')
            biases_2 = cnnu.bias_variable([192], name='local_bias4')
            local4 = tf.nn.relu(tf.matmul(local3, weights_2) + biases_2, name='local4')

            # linear layer(WX + b),
            # We don't apply softmax here because
            # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
            # and performs the softmax internally for efficiency.
            weights_final = cnnu.weight_variable([192, NUM_CLASSES], name='final_fc_weights')
            biases_final = cnnu.bias_variable([NUM_CLASSES], name='final_fc_bias')
            # get only the last part of the output node since it contains also the scope
            softmax_linear = tf.add(tf.matmul(local4, weights_final), biases_final,
                                    name=self.output_node_name.split('/')[1])

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
        test_images, test_cls, test_labels = cifar10_processing.load_test_data()
        # create an instance of dataset class
        self._dataset = DataSet(images, labels, one_hot=True, reshape=False)
        self.test_data = (test_images, test_labels)
        self._train_input_placeholder, self._output_training_node, _ = self._inference(training=True)
        self._input_placeholder, self._output_placeholder, self._label_placeholder = self._inference(training=False)
        # create a global step
        # First create a TensorFlow variable that keeps track of the number of optimization iterations performed so far.
        # we want to save this variable with all the other TensorFlow variables in the checkpoints.
        # Note that trainable=False which means that TensorFlow will not try to optimize this variable.
        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)
        self._loss_node = self._loss(self._output_training_node, self._label_placeholder)
        self._accuracy_node = self._accuracy(self._output_placeholder, self._label_placeholder)
        self._train_step_node = self._train(self._loss_node, global_step=global_step)

    def train(self):
        """
        train the network
        export checkpoints and the metagraph description
        """
        # create folder if it does not exist
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        # Try to restore last checkpoint
        saver = tf.train.Saver()
        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_path)

            # Try and load the data in the checkpoint.
            saver.restore(self._sess, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint. Initializing variables instead.")
            self._sess.run(tf.global_variables_initializer())

        # training iterations
        for i in range(STEPS + 1):
            batch = self._dataset.next_batch(BATCH_SIZE)
            self._sess.run(self._train_step_node,
                           feed_dict={self._train_input_placeholder: batch[0], self._label_placeholder: batch[1]})
            if i % 500 == 0:
                # run the accuracy node
                acc = self._sess.run(self._accuracy_node,
                                     feed_dict={self._input_placeholder: self.test_data[0],
                                                self._label_placeholder: self.test_data[1]})
                str_to_print = "Iteration " + str(i) + ", Acc " + str(acc)
                # log to file
                logging.info(str_to_print)
                saver.save(self._sess, self.checkpoint_prefix, meta_graph_suffix='pb')
        self._save()

    def _accuracy(self, output_node, label_placeholder):
        """
        Get the output node and attach to it the accuracy node
        :param output_node: the output of the net
        :param label_placeholder:
        :return: the accuracy node
        """
        correct_prediction = tf.equal(tf.argmax(output_node, dimension=1), tf.argmax(label_placeholder, dimension=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _save(self):
        saver = tf.train.Saver()
        # export checkpoint variables
        saver.save(self._sess, self.checkpoint_prefix, meta_graph_suffix='pb')
        # export the metagraph, first need to obtain the file name of the meta graph from the total path defined as
        # property
        metagraph_filename = self.metagraph_path.split('/')[len(self.metagraph_path.split('/')) - 1]
        # extract the subgraph
        # graph_to_save = tf.graph_util.extract_sub_graph(self._sess.graph.as_graph_def(),'network/'+self.output_node_name)
        graph_to_save = strip_unused_lib.strip_unused(self._sess.graph.as_graph_def(), [self.input_placeholder_name,
                                                                                        self.label_placeholder_name],
                                                      [self.output_node_name],
                                      dtypes.float32.as_datatype_enum)
        tf.train.write_graph(graph_to_save, self.checkpoint_path, metagraph_filename)

