import torch

from torch import autograd


class LinearFA(autograd.Function):
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None, bias_fa=None):
        context.save_for_backward(input, weight, weight_fa, bias, bias_fa)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias, bias_fa = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = grad_bias_fa = None

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

        return grad_input, grad_weight, grad_weight_fa, grad_bias, grad_bias_fa


class LinearDFA(autograd.Function):
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_dfa, bias=None, bias_dfa=None, output_grad=None):
        context.save_for_backward(input, weight, weight_dfa, bias, bias_dfa, output_grad)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_dfa, bias, bias_fda = context.saved_tensors
        grad_input = grad_weight = grad_weight_dfa = grad_bias = grad_bias_dfa = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_dfa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias, grad_bias_fa, grad_output