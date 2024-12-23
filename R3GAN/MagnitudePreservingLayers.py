import torch
import torch.nn as nn
import math
from torch_utils.ops import bias_act

def Normalize(x, Dimensions=None, ε=1e-4):
    if Dimensions is None:
        Dimensions = list(range(1, x.ndim))
    Norm = torch.linalg.vector_norm(x, dim=Dimensions, keepdim=True, dtype=torch.float32)
    Norm = torch.add(ε, Norm, alpha=math.sqrt(Norm.numel() / x.numel()))
    return x / Norm.to(x.dtype)

def LeakyReLU(x):
    return bias_act.bias_act(x, None, act='lrelu', gain=math.sqrt(2 / (1 + 0.2 ** 2)))
    
class WeightNormalizedConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, EnablePadding, KernelSize):
        super(WeightNormalizedConvolution, self).__init__()
        self.Groups = Groups
        self.EnablePadding = EnablePadding
        self.Weight = nn.Parameter(torch.randn(OutputChannels, InputChannels // Groups, *KernelSize))

    def forward(self, x, Gain=1):
        w = self.Weight.to(torch.float32)
        w = Normalize(w)
        w = w * (Gain / math.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,) if self.EnablePadding else 0, groups=self.Groups)

    def NormalizeWeight(self):
        self.Weight.copy_(Normalize(self.Weight.detach()))
        
def Convolution(InputChannels, OutputChannels, KernelSize, Groups=1):
    return WeightNormalizedConvolution(InputChannels, OutputChannels, Groups, True, [KernelSize, KernelSize])

def Linear(InputDimension, OutputDimension):
    return WeightNormalizedConvolution(InputDimension, OutputDimension, 1, False, [])

class SpatialExtentCreator(nn.Module):
    def __init__(self, OutputChannels):
        super(SpatialExtentCreator, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        
    def forward(self, x):
        return Normalize(self.Basis).view(1, -1, 4, 4) * x.view(x.shape[0], -1, 1, 1)
    
    def NormalizeWeight(self):
        self.Basis.copy_(Normalize(self.Basis.detach()))
    
class SpatialExtentRemover(nn.Module):
    def __init__(self, InputChannels):
        super(SpatialExtentRemover, self).__init__()
        
        self.Basis = WeightNormalizedConvolution(InputChannels, InputChannels, InputChannels, False, [4, 4])
        
    def forward(self, x):
        return self.Basis(x).view(x.shape[0], -1)