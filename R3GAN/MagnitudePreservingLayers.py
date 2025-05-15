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
    
def CosineAttention(x, Heads, QKVLayer, ProjectionLayer):
    y = QKVLayer(x)
    y = y.reshape(y.shape[0], Heads, -1, 3, y.shape[2] * y.shape[3])
    q, k, v = Normalize(y, Dimensions=2).unbind(3)
    w = torch.einsum('nhcq,nhck->nhqk', q, k / math.sqrt(q.shape[2])).softmax(dim=3)
    y = torch.einsum('nhqk,nhck->nhcq', w, v)
    return ProjectionLayer(y.reshape(x.shape[0], -1, x.shape[2], x.shape[3]))
    
class LeakyReLU(nn.Module):
    def __init__(self, α=0.2):
        super(LeakyReLU, self).__init__()
        
        self.α = α
        self.Gain = 1 / math.sqrt(((1 + α ** 2) - (1 - α) ** 2 / math.pi) / 2)

    def forward(self, x):
        return bias_act.bias_act(x, None, act='lrelu', alpha=self.α, gain=self.Gain)

class BoundedParameter(nn.Module):
    def __init__(self, Dimension, Bound=1):
        super(BoundedParameter, self).__init__()
        
        self.Value = nn.Parameter(torch.zeros(Dimension))
        self.Bound = Bound
        
    def forward(self):
        return self.Bound * torch.tanh(self.Value / self.Bound)

class NormalizedWeight(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, KernelSize, Centered):
        super(NormalizedWeight, self).__init__()
        
        self.Centered = Centered
        self.Weight = nn.Parameter(torch.randn(OutputChannels, InputChannels // Groups, *KernelSize))
        
    def Evaluate(self, w):
        if self.Centered:
            w = w - torch.mean(w, axis=list(range(1, w.ndim)), keepdim=True)
        return Normalize(w)
        
    def forward(self):
        return self.Evaluate(self.Weight.to(torch.float32))
    
    def NormalizeWeight(self):
        pass
        # self.Weight.copy_(self.Evaluate(self.Weight.detach()))

class WeightNormalizedConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, EnablePadding, KernelSize, Centered):
        super(WeightNormalizedConvolution, self).__init__()
        
        self.Groups = Groups
        self.EnablePadding = EnablePadding
        self.Weight = NormalizedWeight(InputChannels, OutputChannels, Groups, KernelSize, Centered)

    def forward(self, x, Gain=1):
        w = self.Weight()
        w = w * (Gain / math.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        return nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,) if self.EnablePadding else 0, groups=self.Groups)
        
def Convolution(InputChannels, OutputChannels, KernelSize, Groups=1, Centered=False):
    return WeightNormalizedConvolution(InputChannels, OutputChannels, Groups, True, [KernelSize, KernelSize], Centered)

def Linear(InputDimension, OutputDimension, Centered=False):
    return WeightNormalizedConvolution(InputDimension, OutputDimension, 1, False, [], Centered)

class BiasedPointwiseConvolutionWithModulation(nn.Module):
    def __init__(self, InputChannels, OutputChannels, EmbeddingDimension, Centered=False):
        super(BiasedPointwiseConvolutionWithModulation, self).__init__()
        
        self.Weight = NormalizedWeight(InputChannels + 1, OutputChannels, 1, [1, 1], Centered)
        
        if EmbeddingDimension is not None:
            self.EmbeddingLayer = Linear(EmbeddingDimension, InputChannels, Centered)
            self.EmbeddingGain = nn.Parameter(torch.zeros([]))
        
    def forward(self, x, c, Gain=1, BiasGain=1):
        w = self.Weight()
        w = w / math.sqrt(w[0].numel())
        b = w[:, -1, :, :].view(-1) * BiasGain
        w = w[:, :-1, :, :] * Gain
        if hasattr(self, 'EmbeddingLayer'):
            c = self.EmbeddingLayer(c, Gain=self.EmbeddingGain) + 1
            x = x * c.view(c.shape[0], -1, 1, 1).to(x.dtype)
        return nn.functional.conv2d(x, w.to(x.dtype), b.to(x.dtype))
    
class SpatialExtentCreator(nn.Module):
    def __init__(self, OutputChannels):
        super(SpatialExtentCreator, self).__init__()
        
        self.Basis = NormalizedWeight(1, OutputChannels, 1, [8, 8], False)
        
    def forward(self, x):
        return self.Basis().view(1, -1, 8, 8) * x.view(x.shape[0], -1, 1, 1)
    
class SpatialExtentRemover(nn.Module):
    def __init__(self, InputChannels):
        super(SpatialExtentRemover, self).__init__()
        
        self.Basis = WeightNormalizedConvolution(InputChannels, InputChannels, InputChannels, False, [8, 8], False)
        
    def forward(self, x, Gain):
        return self.Basis(x, Gain=Gain).view(x.shape[0], -1)
    
class ClassEmbedder(nn.Module):
    def __init__(self, NumberOfClasses, EmbeddingDimension):
        super(ClassEmbedder, self).__init__()
        
        self.Weight = NormalizedWeight(EmbeddingDimension, NumberOfClasses, 1, [], False)
    
    def forward(self, x):
        return x @ self.Weight().to(x.dtype)