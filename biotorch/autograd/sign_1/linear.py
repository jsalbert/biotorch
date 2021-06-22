import math
import torch

from torch import autograd

"""
Method from How Important Is Weight Symmetry in Backpropagation? (https://arxiv.org/pdf/1510.05067.pdf)

Uniform Sign-concordant Feedbacks (uSF):
Backward Weights = sign(W)

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
        # Gradient input
        if context.needs_input_grad[0]:
            # We use the sign of the weights to compute the gradients
            # To avoid Exploding Gradients, we scale the sign of the weights by a scaling factor (https://arxiv.org/pdf/1811.03567.pdf)
            # scaling_factor = 1 / math.sqrt(weight.size()[0])
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            grad_input = grad_output.mm(torch.sign(weight) * scaling_factor)
            # To avoid Exploding Gradients, we apply BN + BM
            # Batch Normalization
            # grad_input = torch.nn.functional.batch_norm(grad_input,
            #                                             torch.mean(grad_input, axis=0),
            #                                             torch.var(grad_input, axis=0))
            # Batch Manhattan
            # grad_input = torch.sign(grad_input)
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        # Gradient bias
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
