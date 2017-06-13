echo "running"
python workflow.py --quantize False --evaluate True --train False --module_name CNNs.mnist_models.2conv_2fc --class_name Mnist2Conv2Fc
echo "second net"
python workflow.py --quantize False --evaluate True --train False --module_name CNNs.mnist_models.3fc --class_name Mnist3Fc
echo "third net"
python workflow.py --quantize False --evaluate True --train False --module_name CNNs.mnist_models.big_2conv_3fc --class_name Mnist2Conv3Fc
echo "4 net"
python workflow.py --quantize False --evaluate True --train False --module_name CNNs.mnist_models.big_conv_small_fc --class_name MnistBigConvSmallFc
echo "5 net"
python workflow.py --quantize False --evaluate True --train False --module_name CNNs.mnist_models.small_conv_big_fc --class_name SmallConvBigFc

