import torch

from torch import autograd


class Conv2dGrad(autograd.Function):
    @staticmethod
    def forward(context, input, weight, weight_fa, bias, bias_fa, stride, padding, dilation, groups):
        context.stride, context.padding, context.dilation, context.groups = stride, padding, dilation, groups
        context.save_for_backward(input, weight, weight_fa, bias, bias_fa)
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
        input, weight, weight_fa, bias, bias_fa = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = grad_bias_fa = None

        # Gradient input
        if context.needs_input_grad[0]:
            # Use the FA constant weight matrix to compute the gradient
            grad_input = torch.nn.grad.conv2d_input(input_size=input.shape,
                                                    weight=weight_fa,
                                                    grad_output=grad_output,
                                                    stride=context.stride,
                                                    padding=context.padding,
                                                    dilation=context.dilation,
                                                    groups=context.groups)

        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input=input,
                                                      weight_size=weight_fa.shape,
                                                      grad_output=grad_output,
                                                      stride=context.stride,
                                                      padding=context.padding,
                                                      dilation=context.dilation,
                                                      groups=context.groups)

        # Gradient bias
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).sum(2).sum(1)

        # Return the same number of parameters
        return grad_input, grad_weight, grad_weight_fa, grad_bias, grad_bias_fa, None, None, None, None
