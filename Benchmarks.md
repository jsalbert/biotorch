# Benchmarks

## MNIST & Fashion-MNIST
The training set is split into a 60k training and 10k validation partitions. The model with best validation accuracy is then benchmarked with the same validation set of 10k samples. The model used to compare is LeNet MNIST.
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
The models used to compare are LeNet CIFAR10, ResNet-20 and ResNet-56. The configuration files attached contain the exact hyperparameters used per method. 
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




