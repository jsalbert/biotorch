<h3 align="center">
    <img width="580" alt="BioTorch" src="https://user-images.githubusercontent.com/17982112/121555300-2e01ee80-ca13-11eb-878d-a0f7e8b20401.png">
</h3>

<h3 align="center">
    <p>BioTorch is a PyTorch framework specializing in biologically plausible learning algorithms</p>
</h3>

---
[![Build Status](https://app.travis-ci.com/jsalbert/biotorch.svg?token=961VyHzz93LuqWShsXDX&branch=main)](https://app.travis-ci.com/jsalbert/biotorch)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


BioTorch Provides:

🧠 &nbsp; Implementations of layers, models and biologically-motivated learning algorithms. It allows to load existing state-of-the-art models, easy creation of custom models and automatic conversion of existing models.

🧠 &nbsp; A framework to train, evaluate and benchmark different biologically plausible learning algorithms in a selection of datasets. It is focused on the principles of PyTorch design and research reproducibility. Configuration files that include the choice of a fixed seed and deterministic math and CUDA operations are provided. 

🧠 &nbsp; A place of colaboration, ideas sharing and discussion.  

## Methods Supported

### Feedback Alignment 

| Name  | Mode | Official Implementations|
| :---:         |     :---:      | :---:      |
| [Feedback Alignment](https://arxiv.org/abs/1411.0247)    | `'fa'`     | N/A |
| [Direct Feedback Alignment](https://arxiv.org/abs/1609.01596)    |   `'dfa'`     |[[Torch]](https://github.com/anokland/dfa-torch) |
| [Sign Symmetry](https://arxiv.org/pdf/1510.05067.pdf) | `['usf', 'brsf', 'frsf']`  | [[PyTorch]](https://github.com/willwx/sign-symmetry)|

## Metrics Supported

Layer Weight Alignment            |  Layer Weight Norm Ratio
:-------------------------:|:-------------------------:
![angles_adam](https://github.com/jsalbert/biotorch/blob/main/figures/fa_angles_resnet_56_adam.png?raw=true)  |  ![weight_norm](https://github.com/jsalbert/biotorch/blob/main/figures/fa_weights_resnet_56_adam.png?raw=true)


## Quick Tour

### Create a Feedback Aligment (FA) ResNet-18 model

```python
from biotorch.models.fa import resnet18
model = resnet18()
```

### Create a custom model with uSF layers

```python
import torch.nn.functional as F
from biotorch.layers.usf import Conv2d, Linear

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv1 = Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.fc = Linear(in_features=256, out_features=10)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.avg_pool2d(out, out.size()[3])
    return self.fc(out)
    
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

If you want the experiment to be reproducible, check that you have specified a seed and the parameter `deterministic`is set to true in the configuration file yaml. That will apply all the [PyTorch reproducibility steps](https://pytorch.org/docs/stable/notes/randomness.html). 
If you are running your experiment on GPU add the extra environment variable [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility).

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark.py --config benchmark_configs/mnist/fa.yaml
```

Click [here](https://github.com/jsalbert/biotorch/blob/main/configuration_file.md) to learn more about the configuration file API. 


### Run an experiment on a Colab Notebook

- [Benchmark with configuration file](https://github.com/jsalbert/biotorch/blob/main/notebooks/accuracy_benchmark/benchmark_with_config.ipynb)

- [Benchmark with custom model](https://github.com/jsalbert/biotorch/blob/main/notebooks/accuracy_benchmark/benchmark_custom_model.ipynb)

- [Plot alignment metrics](https://github.com/jsalbert/biotorch/blob/main/notebooks/accuracy_benchmark/metrics_visualizations.ipynb)


## Installation

We are hosted in [PyPI](https://pypi.org/), you can install the library using pip:

```bash
pip install biotorch
```

Or from source:

```bash
git clone https://github.com/jsalbert/biotorch.git
cd biotorch
script/setup
```

## Benchmarks

[MNIST](https://github.com/jsalbert/biotorch/blob/main/Benchmarks.md#mnist--fashion-mnist)

[CIFAR-10](https://github.com/jsalbert/biotorch/blob/main/Benchmarks.md#cifar-10)

[ImageNet](https://github.com/jsalbert/biotorch/blob/main/Benchmarks.md#cifar-10)

## Contributing

If you want to contribute to the project please read the [CONTRIBUTING](https://github.com/jsalbert/biotorch/blob/main/CONTRIBUTING.md) section. If you found any bug please don't hesitate to comment in the [Issues](https://github.com/jsalbert/biotorch/issues) section.

## Related paper: Benchmarking the Accuracy and Robustness of Feedback Alignment Algorithms
##### Albert Jiménez Sanfiz, Mohamed Akrout
Backpropagation is the default algorithm for training deep neural networks due to its simplicity, efficiency and high convergence rate. However, its requirements make it impossible to be implemented in a human brain. In recent years, more biologically plausible learning methods have been proposed. Some of these methods can match backpropagation accuracy, and simultaneously provide other extra benefits such as faster training on specialized hardware (e.g., ASICs) or higher robustness against adversarial attacks. While the interest in the field is growing, there is a necessity for open-source libraries and toolkits to foster research and benchmark algorithms. In this paper, we present BioTorch, a software framework to create, train, and benchmark biologically motivated neural networks. In addition, we investigate the performance of several feedback alignment methods proposed in the literature, thereby unveiling the importance of the forward and backward weight initialization and optimizer choice. Finally, we provide a novel robustness study of these methods against state-of-the-art white and black-box adversarial attacks.

[Preprint here](https://arxiv.org/abs/2108.13446), feedback welcome! 

Contact: albert@aip.ai

If you use our code in your research, you can cite our paper:

```
@misc{sanfiz2021benchmarking,
      title={Benchmarking the Accuracy and Robustness of Feedback Alignment Algorithms},
      author={Albert Jiménez Sanfiz and Mohamed Akrout},
      year={2021},
      eprint={2108.13446},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

