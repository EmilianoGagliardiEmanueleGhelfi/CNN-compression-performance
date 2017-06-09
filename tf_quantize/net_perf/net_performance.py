"""
takes the output of perf and setup the attribute of the object net_object
"""
import json


class NetPerformance:
    def __init__(self, net_name=None, json_dict=None, quantized=None, size=None, path=None):
        """
        :param net_name is the name of the net
        :param json_dict in the string representing this object
        """
        if json_dict is None:
            self.name = net_name
            self.size = size
            self.quantized = quantized
            self.path = path
        else:
            self.__dict__ = json_dict

    def __str__(self):
        representation = ""
        for key in self.__dict__.keys():
            representation += key + ": " + str(self.__dict__[key]) + "\n"
        return representation

    def serialize(self, output_file):
        f = open(output_file, 'w')
        f.write(self.__str__())
        f.close()

    def add_test_information(self, perf_dict):
        """
        Add to self all the attribute with their value contained in dict
        :param perf_dict: the dictionary containing all the information obtained by linux perf, and the accuracy of the model  
        """
        for key in perf_dict.keys():
            setattr(self, key, perf_dict[key])


