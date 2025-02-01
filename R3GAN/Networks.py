import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .MagnitudePreservingLayers import LeakyReLU, Convolution, Linear, SpatialExtentCreator, SpatialExtentRemover, Normalize

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, Cardinality, ExpansionFactor, KernelSize):
        super(ResidualBlock, self).__init__()
        
        ExpandedChannels = InputChannels * ExpansionFactor
        
        self.LinearLayer1 = Convolution(InputChannels, ExpandedChannels, KernelSize=1)
        self.LinearLayer2 = Convolution(ExpandedChannels, ExpandedChannels, KernelSize=KernelSize, Groups=Cardinality)
        self.LinearLayer3 = Convolution(ExpandedChannels, InputChannels, KernelSize=1)
        
    def forward(self, x, InputGain, ResidualGain):
        y = self.LinearLayer1(x, Gain=InputGain)
        y = self.LinearLayer2(LeakyReLU(y))
        y = self.LinearLayer3(LeakyReLU(y), Gain=ResidualGain)
        
        return x + y
    
class ResidualGroup(nn.Module):
    def __init__(self, InputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize):
        super(ResidualGroup, self).__init__()
        
        self.Layers = nn.ModuleList([ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize) for _ in range(NumberOfBlocks)])
        
    def forward(self, x):
        AccumulatedVariance = torch.ones([]).to(x.device)
        for Layer in self.Layers:
            Alpha = 0.2
            x = Layer(x, InputGain=torch.rsqrt(AccumulatedVariance), ResidualGain=Alpha)
            AccumulatedVariance = AccumulatedVariance + Alpha * Alpha
        
        return x, AccumulatedVariance
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x, Gain):
        x = self.LinearLayer(x, Gain=Gain) if hasattr(self, 'LinearLayer') else x * Gain.to(x.dtype)
        x = self.Resampler(x)
        
        return x
    
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x, Gain):
        x = self.Resampler(x)
        x = self.LinearLayer(x, Gain=Gain) if hasattr(self, 'LinearLayer') else x * Gain.to(x.dtype)
        
        return x
    
class GenerativeBasis(nn.Module):
    def __init__(self, InputDimension, OutputChannels):
        super(GenerativeBasis, self).__init__()
        
        self.Basis = SpatialExtentCreator(OutputChannels)
        self.LinearLayer = Linear(InputDimension, OutputChannels)
        
    def forward(self, x):
        return self.Basis(self.LinearLayer(x))
    
class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = SpatialExtentRemover(InputChannels)
        self.LinearLayer = Linear(InputChannels, OutputDimension)
        
    def forward(self, x, Gain):
        return self.LinearLayer(self.Basis(x, Gain=Gain))
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ExpectedMagnitude, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        self.MainLayers = nn.ModuleList([ResidualGroup(WidthPerStage[x], CardinalityPerStage[x], BlocksPerStage[x], ExpansionFactor, KernelSize) for x in range(len(WidthPerStage))])
        self.TransitionLayers = nn.ModuleList([UpsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.BasisLayer = GenerativeBasis(NoiseDimension + ConditionEmbeddingDimension, WidthPerStage[0])
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, KernelSize=1)
        self.ExpectedMagnitude = ExpectedMagnitude
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = Linear(ConditionDimension, ConditionEmbeddingDimension)
        
        self.DataTypePerStage = [torch.float32 for _ in WidthPerStage]
        
    def forward(self, x, y=None):
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        x = self.BasisLayer(x)
        
        for Layer, Transition, DataType in zip(self.MainLayers[:-1], self.TransitionLayers, self.DataTypePerStage[:-1]):
            x, AccumulatedVariance = Layer(x.to(DataType))
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x.to(self.DataTypePerStage[-1]))
        
        return self.AggregationLayer(Normalize(x * torch.rsqrt(AccumulatedVariance)), Gain=self.ExpectedMagnitude)
    
class Discriminator(nn.Module):
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ExpectedMagnitude, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        self.MainLayers = nn.ModuleList([ResidualGroup(WidthPerStage[x], CardinalityPerStage[x], BlocksPerStage[x], ExpansionFactor, KernelSize) for x in range(len(WidthPerStage))])
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.BasisLayer = DiscriminativeBasis(WidthPerStage[-1], 1 if ConditionDimension is None else ConditionEmbeddingDimension)
        self.ExtractionLayer = Convolution(3 + 1, WidthPerStage[0], KernelSize=1)
        self.ExpectedMagnitude = ExpectedMagnitude
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = Linear(ConditionDimension, ConditionEmbeddingDimension)
            self.EmbeddingGain = 1 / math.sqrt(ConditionEmbeddingDimension)
            
        self.DataTypePerStage = [torch.float32 for _ in WidthPerStage]
        
    def forward(self, x, y=None):
        x = x.to(self.DataTypePerStage[0])
        x = torch.cat([x, self.ExpectedMagnitude * torch.ones_like(x[:, :1])], dim=1)
        x = Normalize(self.ExtractionLayer(x))
        
        for Layer, Transition, DataType in zip(self.MainLayers[:-1], self.TransitionLayers, self.DataTypePerStage[:-1]):
            x, AccumulatedVariance = Layer(x.to(DataType))
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x.to(self.DataTypePerStage[-1]))
        
        x = self.BasisLayer(x, Gain=torch.rsqrt(AccumulatedVariance))
        x = (x * self.EmbeddingLayer(y, Gain=self.EmbeddingGain)).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])
