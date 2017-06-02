"""
takes the output of perf and setup the attribute of the object net_object
"""


def setupNet(perf_output, net_object):
    perf_ouput_list = perf_output.split("\n")
    for line in perf_ouput_list:
        # the last line is empty
        if line != "":
            values = line.split(",")
            attrValue = values[0]
            attrName = values[2]
            setattr(net_object, attrName, attrValue)


class NetPerformance:
    name = None
    accuracy = None
    # test time in sec for each image
    test_time = None

    """
	:param net name is the name of the net
	"""

    def __init__(self, net_name, accuracy, perf_output):
        self.name = net_name
        self.accuracy = accuracy
        setupNet(perf_output, self)

    def __str__(self):
        representation = ""
        for key in self.__dict__.keys():
            representation += key + ": " + str(self.__dict__[key]) + "\n"
        return representation
