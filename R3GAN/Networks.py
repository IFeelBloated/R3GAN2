import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .MagnitudePreservingLayers import LeakyReLU, Convolution, Linear, CosineAttention, SpatialExtentCreator, SpatialExtentRemover, Normalize

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, ChannelsPerHead):
        super(MultiHeadSelfAttention, self).__init__()

        self.QKVLayer = Convolution(InputChannels, HiddenChannels * 3, KernelSize=1)
        self.ProjectionLayer = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.Heads = HiddenChannels // ChannelsPerHead

    def forward(self, x, InputGain, ResidualGain):
        QKVLayer = lambda y: self.QKVLayer(y, Gain=InputGain.view(1, -1, 1, 1))
        ProjectionLayer = lambda y: self.ProjectionLayer(y, Gain=ResidualGain.view(-1, 1, 1, 1))
        
        return x + CosineAttention(x, self.Heads, QKVLayer, ProjectionLayer)

class FeedForwardNetwork(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, ChannelsPerGroup, KernelSize):
        super(FeedForwardNetwork, self).__init__()
        
        self.LinearLayer1 = Convolution(InputChannels, HiddenChannels, KernelSize=1, Centered=True)
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, Centered=True)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        
    def forward(self, x, InputGain, ResidualGain):
        y = self.LinearLayer1(x, Gain=InputGain.view(1, -1, 1, 1))
        y = self.LinearLayer2(LeakyReLU(y))
        y = self.LinearLayer3(LeakyReLU(y), Gain=ResidualGain.view(-1, 1, 1, 1))
        
        return x + y
    
class ResidualGroup(nn.Module):
    def __init__(self, InputChannels, BlockConstructors):
        super(ResidualGroup, self).__init__()
        
        self.Layers = nn.ModuleList([Block(**Arguments) for Block, Arguments in BlockConstructors])
        self.ParameterizedAlphas = nn.ParameterList([torch.nn.Parameter(torch.zeros(InputChannels)) for _ in range(len(self.Layers))])

    def forward(self, x):
        AccumulatedVariance = torch.ones([]).to(x.device)
        for ParameterizedAlpha, Layer in zip(self.ParameterizedAlphas, self.Layers):
            Alpha = torch.tanh(ParameterizedAlpha)
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
        x = self.LinearLayer(x, Gain=Gain.view(1, -1, 1, 1)) if hasattr(self, 'LinearLayer') else x * Gain.view(1, -1, 1, 1).to(x.dtype)
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
        x = self.LinearLayer(x, Gain=Gain.view(1, -1, 1, 1)) if hasattr(self, 'LinearLayer') else x * Gain.view(1, -1, 1, 1).to(x.dtype)
        
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
        return self.LinearLayer(self.Basis(x, Gain=Gain.view(-1, 1, 1, 1)))
    
def BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead):
    ResidualGroups = []
    for Width, Blocks in zip(WidthPerStage, BlocksPerStage):
        BlockConstructors = []
        for BlockType in Blocks:
            if BlockType == 'FFN':
                BlockConstructors += [(FeedForwardNetwork, dict(InputChannels=Width, HiddenChannels=round(Width * FFNWidthRatio), ChannelsPerGroup=ChannelsPerConvolutionGroup, KernelSize=KernelSize))]
            elif BlockType == 'Attention':
                BlockConstructors += [(MultiHeadSelfAttention, dict(InputChannels=Width, HiddenChannels=round(Width * AttentionWidthRatio), ChannelsPerHead=ChannelsPerAttentionHead))]
        ResidualGroups += [ResidualGroup(Width, BlockConstructors)]
    return ResidualGroups
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead))
        self.TransitionLayers = nn.ModuleList([UpsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.BasisLayer = GenerativeBasis(NoiseDimension + ConditionEmbeddingDimension, WidthPerStage[0])
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, KernelSize=KernelSize)
        self.Gain = torch.nn.Parameter(torch.ones([]))
        
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

        return self.AggregationLayer(Normalize(x * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1)), Gain=self.Gain)

class Discriminator(nn.Module):
    def __init__(self, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead))
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.BasisLayer = DiscriminativeBasis(WidthPerStage[-1], 1 if ConditionDimension is None else ConditionEmbeddingDimension)
        self.ExtractionLayer = Convolution(3 + 1, WidthPerStage[0], KernelSize=KernelSize)
        self.Bias = torch.nn.Parameter(torch.zeros([]))
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = Linear(ConditionDimension, ConditionEmbeddingDimension)
            self.EmbeddingGain = 1 / math.sqrt(ConditionEmbeddingDimension)
            
        self.DataTypePerStage = [torch.float32 for _ in WidthPerStage]
        
    def forward(self, x, y=None):
        x = x.to(self.DataTypePerStage[0])
        x = Normalize(self.ExtractionLayer(torch.cat([x, math.sqrt(3) * self.Bias.to(x.dtype) * torch.ones_like(x[:, :1])], dim=1)))
        
        for Layer, Transition, DataType in zip(self.MainLayers[:-1], self.TransitionLayers, self.DataTypePerStage[:-1]):
            x, AccumulatedVariance = Layer(x.to(DataType))
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x.to(self.DataTypePerStage[-1]))
        
        x = self.BasisLayer(x, Gain=torch.rsqrt(AccumulatedVariance))
        x = (x * self.EmbeddingLayer(y, Gain=self.EmbeddingGain)).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])
