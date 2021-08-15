<h3 align="center">
    <img width="580" alt="BioTorch" src="https://user-images.githubusercontent.com/17982112/121555300-2e01ee80-ca13-11eb-878d-a0f7e8b20401.png">
</h3>

<h3 align="center">
    <p>BioTorch is a PyTorch framework specialized in biologically plausible learning algorithms</p>
</h3>

---
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

🧠 Provides implementations of layers, models and biologically plausible learning algorithms. Its aim is to build a model hub with the state-of-the-art models and methods in the field.

🧠 Provides a framework to benchmark and compare different biologically plausible learning algorithms in different datasets. 

🧠 Provides a place of colaboration, ideas sharing and discussion.  

## Methods Supported

| Name  | Mode | Official Implementations|
| :---         |     :---      | :---      |
| [Feedback Alignment](https://arxiv.org/abs/1411.0247)    | `'fa'`     |[]|
| [Direct Feedback Alignment](https://arxiv.org/abs/1609.01596)    |   `'dfa'`     |[[Torch]](https://github.com/anokland/dfa-torch) |
| Sign Symmetry([[1]](https://arxiv.org/pdf/1510.05067.pdf), [[2]](https://arxiv.org/abs/1811.03567))    | `['usf', 'brsf', 'frsf']`  | [[PyTorch]](https://github.com/willwx/sign-symmetry)|

There is a branch implementing 
[Weight Mirroring](https://arxiv.org/abs/1904.05391)     |  `'weight_mirroring'` | [[Python]](https://github.com/makrout/Deep-Learning-without-Weight-Transport) | 

## How to use?

### Create a Feedback Aligment (FA) ResNet-18 model

```python
from biotorch.models.fa import resnet18
model = resnet18()
```

### Create a custom model with uSF layers

```python
import torch.nn.functional as F
from biotorch.layers.usf import Conv2d

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(1, 20, 5)
        self.conv2 = Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
model = Model()
```

### Automatically convert AlexNet to use the "frSF" algorithm

```python
from torchvision.models import alexnet
from biotorch.module.biomodule import BioModule

model = BioModule(module=alexnet(), mode='frsf')
```

## Benchmarks

### MNIST & Fashion-MNIST
The training set is split into a 60k training and 10k validation partitions. The model with best validation accuracy is then benchmarked with the same validation set of 10k samples. The model used to compare is LeNet MNIST.
The Top-1 Classification Error Rate is shown in the table.

| Algorithm | MNIST | Fashion MNIST |
|-----------|-------|---------------|
| BP        | 0.84  | 8.88          |
| FA        | 1.91  | 12.77         |
| uSF       | 0.83  | 9.22          |
| brSF      | 0.79  | 9.31          |
| frSF      | 0.92  | 9.27          |
| DFA       | 1.71  | 12.76         |

### CIFAR 10

The training set is split into a 45k training and 5k validation partitions. The model with best validation accuracy is then benchmarked with the testing set of 10k samples as in [He, Kaiming, et al.](https://arxiv.org/abs/1512.03385). 
The models used to compare are LeNet CIFAR10, ResNet-20 and ResNet-56. The configuration files attached contain the exact hyperparameters used per method. 
The Top-1 Classification Error Rate is shown in the table.

| Algorithm | LeNet | LeNet (Adam) | ResNet-20 | ResNet-20 (Adam) | ResNet-56 (Adam) |
|-----------|-------|--------------|-----------|------------------|------------------|
| BP        | 14.52 | 16.37        | 9.42      | 10.27            | 7.91             |
| FA        | 44.15 | 36.35        | 36.38     | 29.16            | 33.04            |
| uSF       | 16.81 | 16.34        | 15.01     | 10.56            | 10.59            |
| brSF      | 17.08 | 17.12        | 14.95     | 11.24            | 13.52            |
| frSF      | 16.95 | 16.58        | 16.94     | 11.29            | 11.5             |
| DFA       | 52.7  | 36.29        | 39.35     | 37.7             | 35.44            |


### ImageNet

