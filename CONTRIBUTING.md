# Contributing to BioTorch

We appreciate all contributions. If you are interested in contributing to BioTorch, there are many ways to help out.

It helps the project if you could:

- Report issues you're facing.
- Give a 👍 on issues that others reported and that are relevant to you.
- Answering queries on the issue tracker, investigating bugs are very valuable contributions to the project.


You would like to improve the documentation. This is no less important than improving the library itself! If you find a typo in the documentation, do not hesitate to submit a GitHub pull request.

If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

**Layers**: 
- Add a new layer for the methods in [layers](https://github.com/jsalbert/biotorch/tree/main/biotorch/layers)
- If the layer is changing the backward pass, add a new [autograd function](https://github.com/jsalbert/biotorch/tree/main/biotorch/autograd)
- Add the layer to the layer utils [function](https://github.com/jsalbert/biotorch/blob/main/biotorch/layers/utils.py)

**Methods**: 
- Add a new method folder and implement the layers (At least Linear and Conv2D) in [layers](https://github.com/jsalbert/biotorch/tree/main/biotorch/layers) with the method name and implement the method
- If the method is changing the backward pass, add a new [autograd function](https://github.com/jsalbert/biotorch/tree/main/biotorch/autograd)
- Add the method to the layer utils [function](https://github.com/jsalbert/biotorch/blob/main/biotorch/layers/utils.py)
- Add the method to the [models folder](https://github.com/jsalbert/biotorch/tree/main/biotorch/models)

## License

By contributing to BioTorch, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
