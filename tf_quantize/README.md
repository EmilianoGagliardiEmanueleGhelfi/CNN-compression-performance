# Tensorflow Quantization

Quantization and evaluation done with quantize tool of tensorflow.

### Perf tool 
In order to setup perf tools (after having installed it)

```
sudo nano /proc/sys/kernel/perf_event_paranoid
Change the value to -1
```

## How to run

- Install tensorflow from sources
- build the quantization tool
- build a neural network following the pattern of abstract class [ToBeQuantized](https://github.com/EmilianoGagliardiEmanueleGhelfi/CNN-compression-performance/blob/master/tf_quantize/pattern/pattern.py)
- run workflow.py with the class as argument

Example:

```
python workflow.py --quantize True --evaluate True --train True --module_name CNNs.mnist_models.2conv_2fc --class_name Mnist2Conv2Fc
```

(Actually it is not very user friendly, sorry)
