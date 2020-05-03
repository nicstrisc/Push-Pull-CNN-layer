# Push-Pull layer for enhanced robustness of ConvNets [[Paper]](https://link.springer.com/article/10.1007/s00521-020-04751-8)
This is the official repository of the Push-Pull layer implementation. The Push-Pull layer can replace convolutional layers in a network (especially in the first layers of ConvNet architectures) and improves the robustness of the network to common corruptions and pertubation not seen during the training. It is inspired by an inhibition phenomenon that happens in the area V1 of the visual cortex, called Push-Pull inhibition.

The Push-Pull layer was proposed in the following paper:

N. Strisciuglio, M. Lopez-Antequera, N. Petkov, __Enhanced robustness of convolutional networks with a push–pull inhibition layer__, _Neural Computing and Applications, 2020_, doi: 10.1007/s00521-020-04751-8



## Data sets
In [1], experiments on the CIFAR-10 were reported. The original CIFAR-10 is available from the official repository of the CIFAR Institute and it is downloaded within the Python scripts available in this repository.

The CIFAR-C and CIFAR-P data sets are available from [https://zenodo.org/record/2535967#.Xq7nNp9fhhE](https://zenodo.org/record/2535967#.Xq7nNp9fhhE). Please, download and un-tar them yourself.

## Getting started
This repository containes the following folders:

* __datasets__: loaders for the data sets
* __densenet__: implementation of DenseNet with options for the push-pull layer
* __models__: trained models (with and without push-pull layers) 
* __pushpull__: implementation of the Push-Pull layer
* __resnet__: implementation of resnet with options for the push-pull layer
* __results__: results on CIFAR, CIFAR-C and CIFAR-P data sets
* __utils__: utilities 

To train a ResNet-20 model on CIFAR with a push-pull layer replacing the first convolutional layer, run:

```bash
python -m train --arch resnet --name resnet-20-pp --layers 20 --pushpull --print-freq 50
```

To test a ResNet-20 with a push-pull layer on the CIFAR-C data, run (note that the script loads the pre-trained models from the 'models' folder - you can change it in the code of the test_corruption.py file):
```bash
python -m test_corruption --arch resnet --name resnet-20-pp --pushpull --layers 20 --corrupted-data-dir /path/to/CIFAR-C/root/folder/ 
```

To test a ResNet-20 with a push-pull layer on the CIFAR-P data, run (note that the script loads the pre-trained models from the 'models' folder - you can change it in the code of the test_perturbation.py file):
```bash
python -m test_perturbation --arch resnet --name resnet-20-pp --pushpull --layers 20 --perturbed-datadir /path/to/CIFAR-P/root/folder/
```

## References
If you use the Push-Pull layer (this or other implementations), please cite the original paper: 

[1] N. Strisciuglio, M. Lopez-Antequera, N. Petkov, __Enhanced robustness of convolutional networks with a push–pull inhibition layer__, _Neural Computing and Applications, 2020_, doi: 10.1007/s00521-020-04751-8

	article{StrisciuglioNCAA2020,
	title = {Enhanced robustness of convolutional networks with a push–pull inhibition layer},
	author = {Nicola Strisciuglio and Manuel Lopez-Antequera and Nicolai Petkov},
	editor = {Springer},
	year = {2020},
	journal = {Neural Computing and Applications},
	doi={10.1007/s00521-020-04751-8}
	}
	
## Acknowledgements
The development of the Push-Pull layer was partially supported by the EU H2020 research and innovation program, grant no. 688007 (TrimBot2020).
Please, visit the [website of the TrimBot2020 project](http://www.trimbot2020.org).

The code was developed by Manuel Lopez-Antequera and Nicola Strisciuglio and is maintained by Nicola Strisciuglio.
