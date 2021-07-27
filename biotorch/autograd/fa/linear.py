from torch import autograd


class LinearGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the weight_backward matrix of the layer
    """
    @staticmethod
    # Same as reference linear function, but with additional weight tensor for backward
    def forward(context, input, weight, weight_backward, bias=None, bias_backward=None):
        context.save_for_backward(input, weight, weight_backward, bias, bias_backward)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_backward, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_weight_backward = grad_bias = grad_bias_backward = None
        # Gradient input
        if context.needs_input_grad[0]:
            # Use the weight_backward matrix to compute the gradient
            grad_input = grad_output.mm(weight_backward)
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        # Gradient bias
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_backward, grad_bias, grad_bias_backward
