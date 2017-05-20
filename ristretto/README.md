# Introduction

For real world application, convolutional neural network(CNN) model can take more than 100MB of space and can be computationally too expensive. Therefore, there are multiple methods to reduce this complexity in the state of art. Ristretto is a plug-in to Caffe framework that employs several model approximation methods. For this projects, ﬁrst a CNN model will be trained for Cifar-10 dataset with Caffe, then Ristretto will be use to generate multiple approximated version of the trained model using different schemes. The goal of this projects is comparison of the models in terms of execution performance, model size and cache utilizations in the test or inference phase.

# How to use

- Write your path to ristretto executable in the environment variable RISTRETTOPATH

```
export RISTRETTOPATH=path/to/ristretto/executable
```

- Install valgrind

```
sudo apt-get install valgrind
```

- Create a config.ini file
- The network protofile have to contain the absolute path to the dataset or the relative from the current working directory
- in the solver protofile the path to the network and the snapshot prefix have to be respect to the current working directory

- Run

```
python main.py config.ini
``` 

## Config file
Here's an example of a config.ini file with the supported parameters:

```
[CAFFE]
# True for training, false otherwise
train=True
# Solver path relative to the current working directory
solver_path=models_mnist/solver.prototxt

[RISTRETTO]
# True if you need to compress with these schemes
dynamic_fixed_point=True
minifloat=True
integer_power_of_2_weights=True
# ristretto quantize parameters
iterations=10
error_margin=1

[TEST]
# number of iterations in test phase
iterations=10
# True if need to evaluate perfomances
performance=True
# folder in which the output file of valgrind will be written (relative to the current working direcory)
performance_path = perf
```

# Authors

- Emanuele Ghelfi
- Emiliano Gagliardi
