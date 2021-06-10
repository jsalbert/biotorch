import torch

from torch import autograd

"""
Method from How Important Is Weight Symmetry in Backpropagation? (https://arxiv.org/pdf/1510.05067.pdf)

Fixed Random Magnitude Sign-concordant Feedbacks (frSF):
V = M ◦ sign(W), where M is initialized once and fixed throughout each experiment

"""


# Regular Back-Propagation computing the gradients with the sign of the forward weight matrix
class Conv2dGrad(autograd.Function):
    @staticmethod
    def forward(context, input, weight, weight_backward, bias, bias_backward, stride, padding, dilation, groups):
        context.stride, context.padding, context.dilation, context.groups = stride, padding, dilation, groups
        context.save_for_backward(input, weight, weight_backward, bias, bias_backward)
        output = torch.nn.functional.conv2d(input,
                                            weight,
                                            bias=bias,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_backward, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_bias = grad_weight_backward = grad_bias_backward = None

        # Gradient input
        if context.needs_input_grad[0]:
            # We use the sign of the weights to compute the gradients
            grad_input = torch.nn.grad.conv2d_input(input_size=input.shape,
                                                    weight=torch.sign(weight) * weight_backward,
                                                    grad_output=grad_output,
                                                    stride=context.stride,
                                                    padding=context.padding,
                                                    dilation=context.dilation,
                                                    groups=context.groups)
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input=input,
                                                      weight_size=weight.shape,
                                                      grad_output=grad_output,
                                                      stride=context.stride,
                                                      padding=context.padding,
                                                      dilation=context.dilation,
                                                      groups=context.groups)
        # Gradient bias
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).sum(2).sum(1)

        # We return the same number of parameters
        return grad_input, grad_weight, grad_weight_backward, grad_bias, grad_bias_backward, None, None, None, None