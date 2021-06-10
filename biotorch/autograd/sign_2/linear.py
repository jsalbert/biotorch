import torch

from torch import autograd

"""
Method from How Important Is Weight Symmetry in Backpropagation? (https://arxiv.org/pdf/1510.05067.pdf)

Batchwise Random Magnitude Sign-concordant Feedbacks (brSF):
V = M ◦ sign(W), where M is redrawn after each update of W (i.e., each mini-batch).

"""


class LinearGrad(autograd.Function):
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, bias=None):
        context.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, bias = context.saved_tensors
        grad_input = grad_weight = grad_bias = None
        weight_backward = torch.Tensor(weight.size(), requires_grad=False)
        # Gradient input
        if context.needs_input_grad[0]:
            # We use the sign of the weights to compute the gradients
            grad_input = grad_output.mm(torch.sign(weight) * weight_backward)
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        # Gradient bias
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
