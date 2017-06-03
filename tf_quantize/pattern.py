from abc import ABCMeta, abstractmethod, abstractproperty


class ToBeQuantizedNetwork:
    """
    Definition of the interface of a class that defines a neural network that we want to quantize using the tf_quantize
    quantization tool. The class define the computational graph of the network, the training, and the evaluation,
    implementing the pattern prepare, train, quantize.
    Then also there is the need of implementing a method that should be called after the training of the network to
    export the computational graph and the network model in pb format. That file will be the input of the quantization
    tf_quantize tool.
    """
    __metaclass__ = ABCMeta

    """
    Properties needed to run the test of the quantized network
    """

    @abstractproperty
    def test_iterations(self):
        """
        need to define the iteration used in the test algorithm, in order to make comparable the performance in term
        of test time and cache miss of the network and the quantized network
        """
        pass

    @abstractproperty
    def test_data(self):
        """
        a tuple containing the test data set value and the test data set labels. Need to define it because the workflow
        script runs the test of the quantized network using this values
        """
        pass

    @abstractproperty
    def input_placeholder_name(self):
        """
        The name used to export with tf.collection the input placeholder of the network
        """
        pass

    @abstractproperty
    def label_placeholder_name(self):
        """
        The name used to export with tf.collection the label placeholder of the network
        """
        pass

    @abstractproperty
    def output_node_name(self):
        """
        The name used to export with tf.collection the label placeholder of the network
        """
        pass

    """
    Properties needed to execute the export to pb method
    """

    @abstractproperty
    def checkpoint_prefix(self):
        """
        The path passed to save for saving the session: after the training of the network save the data with saver.save
        """
        pass

    @abstractproperty
    def metagraph_path(self):
        """
        The path to the meta graph obtained by the network: after the network computational graph have been created
        export it with tf.train.write_graph()
        """
        pass

    @abstractproperty
    def output_pb_path(self):
        """
        The path of the file obtained by the export_to_pb method in workflow. This will be the input of the quantization
        tf_quantize tool
        """
        pass

    @abstractproperty
    def output_quantized_graph(self):
        """
        The path to give to the quantization tool where the quantized network will be stored
        """
        pass

    """
    End of properties
    """

    """
    prepare, train
    """

    @abstractmethod
    def prepare(self):
        """
        This method creates the network, with all the nodes needed to do prediction, training and evaluation.
        Also obtains and prepare the data set
        """
        pass

    @abstractmethod
    def train(self):
        """
        This methods contains the training algorithm, that uses the training node defined in prepare
        """
        pass