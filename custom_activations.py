import numpy as np

import torch
from torch.autograd import Variable

import torch.nn.functional as F

from functools import partial

"""
Built-In Activations:
F.elu(input, alpha=1., inplace=False)
F.selu(input, inplace=False)
F.leaky_relu(input, negative_slope=0.01, inplace=False)
F.softplus(input, beta=1, threshold=20)
F.softsign(input)
F.hardtanh(input, min_val=-1., max_val=1., inplace=False) 
F.tanhshrink(input)
F.rrelu(input, lower=1. / 8, upper=1. / 3, training=False, inplace=False) # not too different from some others
"""

def swish(x):
    return x * F.sigmoid(x)

def eswish(x,Beta=1.625):
    return Beta*x*F.sigmoid(x)

def tswish(x):
    return x * F.tanh(x)

def teswish(x,Beta=1.625):
    return Beta*x*F.tanh(x)

def swishleak(x,negative_slope=0.01):
    return ( (x * F.sigmoid(x)) + F.leaky_relu(x,negative_slope=negative_slope) ) / 2

def swishtanhleak(x,negative_slope=0.05):
    return ( (x * F.tanh(x)) + F.leaky_relu(x, negative_slope=negative_slope) ) /2

def tanhleak(x,negative_slope=0.05):
    return ( F.tanh(x) + F.leaky_relu(x, negative_slope=negative_slope) ) /2

def tanh2leak(x,negative_slope=0.05):
    return ( F.tanh(x) + F.tanh(x) + F.leaky_relu(x, negative_slope=negative_slope) ) /3

def tanhsigleak(x,negative_slope=0.1):
    return ( F.tanh(x) + F.sigmoid(x) + F.leaky_relu(x, negative_slope=negative_slope) ) /3

def eswishtanhleak(x,Beta=1.5,negative_slope=0.03):
    return ( (Beta * x * F.tanh(x)) + F.leaky_relu(x, negative_slope=negative_slope) ) /2

def eswishrelu(x,Beta=1.625):
    return F.relu(Beta*x*F.sigmoid(x))

def eswishleak(x,Beta=0.95,negative_slope=0.05):
    return (Beta * (F.sigmoid(x) + F.leaky_relu(x, negative_slope=negative_slope) )) /3

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

def pennington1(x,c1=.5):
    output = torch.exp(-2 * x**2)
    output = torch.sqrt(5* output)
    return c1*(-1 + output)

# can also invert by doing negative and see if that works any better
def pennington2(x,c2=.0005):
    sin1 = torch.sin(2*x)
    cos1 = torch.cos(3*x/2)
    exp1 = float(np.exp(-2))
    exp2 = float(np.exp(-9/8))
    return c2 * (sin1 + cos1 - 2 * exp1 * x - exp2)

################ Simple HyperParameter Adjustments

# leaky_relu default:
# F.leaky_relu(input, negative_slope=0.01, inplace=False)

leaky_relu_ns_006 = partial(F.leaky_relu, negative_slope=0.006)
leaky_relu_ns_02 = partial(F.leaky_relu, negative_slope=0.02)
leaky_relu_ns_08 = partial(F.leaky_relu, negative_slope=0.08)

# eswish default:
# eswish(x,Beta=1.625):
eswish_1_1 = partial(eswish,Beta=1.1)
eswish_1_8 = partial(eswish,Beta=1.8)

# TODO: eswishrelu, swishleak

bidrelmomv2_tv11_m06 = partial(bidrelu_momentum_v2,threshold_value=0.11,momentum=0.06):
bidrelmomv2_tv23_m15 = partial(bidrelu_momentum_v2,threshold_value=0.23,momentum=0.15):
bidrelmomv2_tv32_m18 = partial(bidrelu_momentum_v2,threshold_value=0.32,momentum=0.18):

#########################################################################

# Other TODO and reference

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
F.prelu(input, weight) # need to figure out how to make the weight parameter hold
F.threshold(input, threshold, value, inplace=False)
LReLU?

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