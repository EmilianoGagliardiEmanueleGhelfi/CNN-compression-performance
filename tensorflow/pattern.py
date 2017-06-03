from abc import ABCMeta, abstractmethod, abstractproperty


class ToBeQuantizedNetwork:
    """
    Definition of the interface of a class that defines a neural network that we want to quantize using the tensorflow
    quantization tool. The class define the computational graph of the network, the training, and the evaluation,
    implementing the pattern prepare, train, quantize.
    Then also there is the need of implementing a method that should be called after the training of the network to
    export the computational graph and the network model in pb format. That file will be the input of the quantization
    tensorflow tool.
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

    @abstractproperty
    def accuracy_node_name(self):
        """
        The name used to export with tf.collection the accuracy node of the network
        """
        pass

    """
    Properties needed to execute the export to pb method
    """

    @abstractproperty
    def checkpoint_prefix(self):
        """
        The path passed to save for saving the session
        """
        pass

    @abstractproperty
    def metagraph_path(self):
        """
        The path to the metagraph obtained by the network
        """
        pass

    @abstractproperty
    def output_pb_path(self):
        """
        The path of the file obtained by the export_to_pb method in workflow. This will be the input of the quantization
        tensorflow tool
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

    @abstractmethod
    def export_nodes(self):
        """
        This method exports with tf.collection the input and label placeholders and the output and the accuracy node.
        They will be imported by the workflow script to perform the tests
        """
        pass

    """
    prepare, train, evaluate pattern
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

    @abstractmethod
    def evaluate(self):
        """
        This methods contains the test algorithm, that uses the accuracy node defined in prepare
        """
