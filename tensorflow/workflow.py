"""
This script takes as input a script that defines a cnn, following the prepare, train, evaluate pattern.
"""

import mnist_models.mnist_pattern_implementation as e


def main():
    model = e.MnistNetwork()
    model.prepare()
    model.train()
    print model.evaluate()

    print "Now I'll try to export everything, wish me luck!"
    model.export_to_pb()


if __name__ == '__main__':
    main()
