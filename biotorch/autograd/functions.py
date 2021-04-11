import torch

from torch import autograd


class LinearFA(autograd.Function):
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None
        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)

        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class Conv2dFA(autograd.Function):
    @staticmethod
    def forward(context, input, kernels, kernels_fa, bias, bias_fa, stride, padding):
        context.save_for_backward(input, kernels, kernels_fa, bias, bias_fa)
        output = torch.nn.functional.conv2d(input, kernels, bias=bias, stride=stride, padding=padding)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, kernels, kernels_fa, bias, bias_fa = context.saved_tensors
        grad_input = grad_kernels = grad_kernels_fa = grad_bias = grad_bias_fa = None

        if context.needs_input_grad[0]:
           grad_input = torch.nn.grad.conv2d_input(input.shape, kernels_fa, grad_output)

        if context.needs_input_grad[1]:
            grad_kernels = torch.nn.grad.conv2d_weight(input, kernels_fa.shape, grad_output)

        if context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).sum(2).sum(1)

        # add the input in the stride gradient which is useless
        # return grad_input, grad_kernels, grad_kernels_fa, grad_bias, grad_bias_fa, input, None
        return grad_input, grad_kernels, input, grad_bias, grad_output, None, None
