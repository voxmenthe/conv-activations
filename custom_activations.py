import torch
from torch.autograd import Variable

import torch.nn.functional as F

"""
Built-In Activations:
F.elu(input, alpha=1., inplace=False)
F.selu(input, inplace=False)
F.leaky_relu(input, negative_slope=0.01, inplace=False)
F.rrelu(input, lower=1. / 8, upper=1. / 3, training=False, inplace=False)
F.prelu(input, weight)
F.hardtanh(input, min_val=-1., max_val=1., inplace=False) 
F.threshold(input, threshold, value, inplace=False)
F.glu(input, dim=-1)
F.logsigmoid(input)
F.softsign(input)
LReLU?
"""

def swish(x):
    return x * F.sigmoid(x)

def eswish(x,Beta=1.625):
	return Beta*x*F.sigmoid(x)

def eswishrelu(x,Beta=1.625):
    return F.relu(Beta*x*F.sigmoid(x))

def drelu(x,threshold_value=0.05,output=0):
    return F.threshold(x,threshold_value,output)

def bidrelu(x,threshold_value=0.05,output=0):
    xp = F.threshold(x,threshold_value,output)
    xn = -F.threshold(-x,threshold_value,output)
    return xp+xn

def bidrelu_momentum_v2(x,threshold_value=0.15,momentum=0.05):
    assert threshold_value > 0, "bidrelu threshold_value must be > 0"
    assert threshold_value >= momentum, "bidrelu threshold_value must be >= momentum" 
    
    xp = F.threshold(x,threshold_value,0)
    xp = F.threshold(xp+momentum,threshold_value,0)

    xn = F.threshold(-x,threshold_value,0)+momentum
    xn = -F.threshold(xn,threshold_value,0)
    return xp+xn

def bidrelu_skewed_momentum(x,threshold_value=0.25,momentum=0.05):
    xp = F.threshold(x,threshold_value,0)
    xp = F.threshold(xp+momentum,threshold_value,0)

    xn = -F.threshold(-x,threshold_value,0)
    xn = -F.threshold(-(xn+momentum),threshold_value,0)
    return xp+xn


################ Simple HyperParameter Adjustments
def bidrelmomv2_tv23_m15(x,threshold_value=0.23,momentum=0.15):
    assert threshold_value > 0, "bidrelu threshold_value must be > 0"
    assert threshold_value >= momentum, "bidrelu threshold_value must be >= momentum" 
    
    xp = F.threshold(x,threshold_value,0)
    xp = F.threshold(xp+momentum,threshold_value,0)

    xn = F.threshold(-x,threshold_value,0)+momentum
    xn = -F.threshold(xn,threshold_value,0)
    return xp+xn



# class DReLU(torch.autograd.Function):
#   def forward(self, input):
#     self.save_for_backward(input)
#     return input.clamp(min=0.05)

#   def backward(self, grad_output):
#     input, = self.saved_tensors
#     grad_input = grad_output.clone()
#     grad_input[input < 0.05] = 0.05
#     return grad_input

# drelu = DReLU()



# Other Reference:
"""
def Clamp(x, minval):
    '''
    Clamps Variable x to minval.
    minval <= 0.0
    '''
    return x.clamp(max=0.0).sub(minval).clamp(min=0.0).add(minval) + x.clamp(min=0.0)

class MyReLU(torch.autograd.Function):
  def forward(self, input):
    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input

Class Swish(Function):
    @staticmethod
    def forward(ctx, i):
        result = i*i.sigmoid()
        ctx.save_for_backward(result,i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result,i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result+sigmoid_x*(1-result))

swish= Swish.apply

class Swish_module(nn.Module):
    def forward(self,x):
        return swish(x)
    
swish_layer = Swish_module()
"""