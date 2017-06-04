"""
takes the output of perf and setup the attribute of the object net_object
"""


def setup_net(perf_output, net_object):
    perf_output_list = perf_output.split("\n")
    for line in perf_output_list:
        # the last line is empty
        if line != "":
            values = line.split(",")
            attr_value = values[0]
            attr_name = values[2]
            setattr(net_object, attr_name, attr_value)


class NetPerformance:
    name = None
    accuracy = None
    # test time in sec for each image
    test_time = None

    def __init__(self, net_name, accuracy, perf_output):
        """
        :param net_name is the name of the net
        :param accuracy is the accuracy of the net in testing
        :param perf_output is the string output of perf_tool, to be parsed
        """
        self.name = net_name
        self.accuracy = accuracy
        setup_net(perf_output, self)

    def __str__(self):
        representation = ""
        for key in self.__dict__.keys():
            representation += key + ": " + str(self.__dict__[key]) + "\n"
        return representation
