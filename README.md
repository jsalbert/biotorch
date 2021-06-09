# BioTorch
BioTorch is a PyTorch framework specialized in biologically plausible learning algorithms


🧠 Provides implementations of layers, models and biologically plausible learning algorithms. Its aim is to build a model hub with the state-of-the-art models and methods in the field.

🧠 Provides a framework to benchmark and compare different biologically plausible learning algorithms in different datasets. 

🧠 Provides a place of colaboration, ideas sharing and discussion.  

## Motivation

During learning, the brain modifies synapses to improve behaviour. In the cortex, synapses are embedded within multilayered networks, making it difficult to determine the effect of an individual synaptic modification on the behaviour of the system. The backpropagation algorithm solves this problem in deep artificial neural networks, but historically it has been viewed as biologically problematic. Nonetheless, recent developments in neuroscience and the successes of artificial neural networks have reinvigorated interest in whether backpropagation offers insights for understanding learning in the cortex. The backpropagation algorithm learns quickly by computing synaptic updates using feedback connections to deliver error signals. Although feedback connections are ubiquitous in the cortex, it is difficult to see how they could deliver the error signals required by strict formulations of backpropagation.

_Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton, G. (2020). Backpropagation and the brain. Nature Reviews Neuroscience, 21(6), 335-346._

## Methods Supported

- Feedback Alignment
- Direct Feedback Alignment
- Sign Symmetry
- Weight Mirroring
