# Benchmarks

## MNIST & Fashion-MNIST
The training set is split into a 60k training and 10k validation partitions. The model with best validation accuracy is then benchmarked with the same validation set of 10k samples. The model used to compare is [LeNet MNIST](https://github.com/jsalbert/biotorch/blob/main/biotorch/models/le_net.py#L6). 
Networks were trained with the Stochastic Gradient Descent (SGD) optimizer with a momentum of 0.9, and weight decay of 10^-3. Images were resized to 32x32 prior to being input to the network. We trained with a batch size of 64 for 100 epochs in one GPU. We decreased the initial learning rate by a factor of 10 at the 50th and the 75th epoch.
The Top-1 Classification Error Rate is shown in the table.

| Method | MNIST | Fashion MNIST |
|:--------:|:-------:|:---------------:|
| BP     |  0.91 |           9.2 |
| FA     |   1.7 |         13.06 |
| uSF    |  0.94 |          9.69 |
| brSF   |  0.91 |         10.02 |
| frSF   |  0.97 |          9.61 |
| DFA    |  1.61 |         12.81 |

## CIFAR 10

The training set is split into a 45k training and 5k validation partitions. The model with best validation accuracy is then benchmarked with the testing set of 10k samples as in [He, Kaiming, et al.](https://arxiv.org/abs/1512.03385). 
The models used to compare are [LeNet CIFAR10](https://github.com/jsalbert/biotorch/blob/main/biotorch/models/le_net.py#L41), [ResNet-20](https://github.com/jsalbert/biotorch/blob/main/biotorch/models/small_resnet.py#L115) and [ResNet-56](https://github.com/jsalbert/biotorch/blob/main/biotorch/models/small_resnet.py#L127). The configuration files attached contain the exact hyperparameters used per method. 
The Top-1 Classification Error Rate is shown in the table.

| Method | LeNet | LeNet (Adam) | ResNet-20 | ResNet-20 (Adam) | ResNet-56 (SGD) | ResNet-56 (Adam) |
|:------:|:-----:|:------------:|:---------:|:----------------:|:---------------:|:----------------:|
|   BP   | 14.23 |     15.92    |    8.63   |       10.01      |       8.3       |       7.83       |
|   FA   | 46.69 |     40.67    |   32.16   |       29.59      |      34.88      |       29.23      |
|   DFA  | 54.21 |     37.59    |   45.94   |       32.16      |      38.01      |       32.02      |
|   uSF  | 16.22 |     16.34    |   10.05   |       10.59      |       8.2       |       9.19       |
|  brSF  | 16.02 |     17.08    |   11.02   |       11.08      |       8.69      |       10.13      |
|  frSF  | 16.86 |     16.83    |    11.2   |       11.22      |       9.49      |       10.02      |


## ImageNet

A ResNet-18 network is trained with a batch size of 256 and 2 GPUs for 75 epochs using SGD with a initial learning rate of 0.1. A scheduler decreased the learning rate by a factor of 10 at the 20th, the 40th and the 60th epoch. We used a weight decay of 10^-4 and a momentum of 0.9. For DFA we used Adam with an initial learning rate of 0.001. At training time, a random resized crop of dimensions 224x224 of the original image or its horizontal flip with the per-pixel mean subtracted is used. When testing, the image is resized to 256x256 and then a center crop of 224x224 is used as input to the network.

| Method | ResNet-18 |
|:------:|:---------:|
|   BP   |   30.39   |
|   FA   |   85.25   |
|   DFA  |   82.45   |
|   uSF  |   34.97   |
|  brSF  |   37.21   |
|  frSF  |    36.5   |

