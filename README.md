<h3 align="center">
    <img width="580" alt="BioTorch" src="https://user-images.githubusercontent.com/17982112/121555300-2e01ee80-ca13-11eb-878d-a0f7e8b20401.png">
</h3>

<h3 align="center">
    <p>BioTorch is a PyTorch framework specialized in biologically plausible learning algorithms</p>
</h3>

---
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Build Status](https://app.travis-ci.com/jsalbert/biotorch.svg?token=961VyHzz93LuqWShsXDX&branch=main)](https://app.travis-ci.com/jsalbert/biotorch)

BioTorch Provides:

🧠 Implementations of layers, models and biologically plausible learning algorithms. It allows to load existing state-of-the-art models, easy creation of custom models and automatic conversion of any existing model.

🧠 A framework to train, evaluate and benchmark different biologically plausible learning algorithms in a selection of datasets. It is focused on the principles of PyTorch design and research reproducibility. Configuration files that include the choice of a fixed seed and deterministic math and CUDA operations are provided. 

🧠 A place of colaboration, ideas sharing and discussion.  

## Methods Supported

| Name  | Mode | Official Implementations|
| :---         |     :---      | :---      |
| [Feedback Alignment](https://arxiv.org/abs/1411.0247)    | `'fa'`     |[]|
| [Direct Feedback Alignment](https://arxiv.org/abs/1609.01596)    |   `'dfa'`     |[[Torch]](https://github.com/anokland/dfa-torch) |
| [Sign Symmetry](https://arxiv.org/pdf/1510.05067.pdf) ([[2](https://arxiv.org/abs/1811.03567)])    | `['usf', 'brsf', 'frsf']`  | [[PyTorch]](https://github.com/willwx/sign-symmetry)|

There is a branch implementing 
[Weight Mirroring](https://arxiv.org/abs/1904.05391)     |  `'weight_mirroring'` | [[Python]](https://github.com/makrout/Deep-Learning-without-Weight-Transport) | 


## Quick Tour

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

### Run an experiment on the command line

```bash
python benchmark.py --config benchmark_configs/mnist/fa.yaml
```

If you want the experiment to be reproducible, check that you have specified a seed and the parameter `deterministic`is set to True in the configuration file yaml. That will apply all the [PyTorch reproducibility steps](https://pytorch.org/docs/stable/notes/randomness.html). 
If you are running your experiment on GPU add the extra environment variable [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility).

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark.py --config benchmark_configs/mnist/fa.yaml
```


Click here to learn more about the configuration file API. 


### Run an experiment on a Jupyter Notebook





## Installation





## Benchmarks

## Contributing



