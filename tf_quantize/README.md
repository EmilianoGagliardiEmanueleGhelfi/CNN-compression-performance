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
- build a neural network following the pattern of abstract class ToBeQuantized
- run workflow.py with the class as argument