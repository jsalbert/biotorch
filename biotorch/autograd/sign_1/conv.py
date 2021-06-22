import torch
import math

from torch import autograd

"""
Method from How Important Is Weight Symmetry in Backpropagation? (https://arxiv.org/pdf/1510.05067.pdf)

Uniform Sign-concordant Feedbacks (uSF):
Backward Weights = sign(W)

"""


class Conv2dGrad(autograd.Function):
    @staticmethod
    def forward(context, input, weight, bias, stride, padding, dilation, groups):
        context.stride, context.padding, context.dilation, context.groups = stride, padding, dilation, groups
        context.save_for_backward(input, weight, bias)
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
        input, weight, bias = context.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Gradient input
        if context.needs_input_grad[0]:
            # We use the sign of the weights to compute the gradients
            # To avoid Exploding Gradients, we scale the sign of the weights by a scaling factor (https://arxiv.org/pdf/1811.03567.pdf)
            ws = weight.size()
            # scaling_factor = math.sqrt(2 / (ws[0] * ws[2] * ws[3]))
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            grad_input = torch.nn.grad.conv2d_input(input_size=input.shape,
                                                    weight=torch.sign(weight)*scaling_factor,
                                                    grad_output=grad_output,
                                                    stride=context.stride,
                                                    padding=context.padding,
                                                    dilation=context.dilation,
                                                    groups=context.groups)


            # To avoid Exploding Gradients, we apply BN + BM
            # Batch Normalization (Axis = 1 are the conv layer channels)
            # grad_input = torch.nn.functional.batch_norm(grad_input,
            #                                             torch.mean(grad_input, axis=(0, 2, 3)),
            #                                            torch.var(grad_input, axis=(0, 2, 3), unbiased=False))
            # Batch Manhattan (Only use the sign)
            # grad_input = torch.sign(grad_input)
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
        return grad_input, grad_weight, grad_bias, None, None, None, None
