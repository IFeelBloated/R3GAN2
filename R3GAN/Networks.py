import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler, InplaceUpsampler, InplaceDownsampler
from .MagnitudePreservingLayers import LeakyReLU, Convolution, Linear, BiasedPointwiseConvolution, CosineAttention, SpatialExtentCreator, SpatialExtentRemover

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, ChannelsPerHead):
        super(MultiHeadSelfAttention, self).__init__()

        self.QKVLayer = BiasedPointwiseConvolution(InputChannels, HiddenChannels * 3, Centered=True)
        self.ProjectionLayer = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.Heads = HiddenChannels // ChannelsPerHead

    def forward(self, x, InputGain, ResidualGain):
        QKVLayer = lambda y: self.QKVLayer(y, Gain=InputGain.view(1, -1, 1, 1))
        ProjectionLayer = lambda y: self.ProjectionLayer(y, Gain=ResidualGain.view(-1, 1, 1, 1))
        
        return x + CosineAttention(x, self.Heads, QKVLayer, ProjectionLayer)

class FeedForwardNetwork(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, ChannelsPerGroup, KernelSize):
        super(FeedForwardNetwork, self).__init__()
        
        self.LinearLayer1 = BiasedPointwiseConvolution(InputChannels, HiddenChannels, Centered=True)
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, Centered=True)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.NonLinearity = LeakyReLU()
        
    def forward(self, x, InputGain, ResidualGain):
        y = self.LinearLayer1(x, Gain=InputGain.view(1, -1, 1, 1))
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y), Gain=ResidualGain.view(-1, 1, 1, 1))
        
        return x + y
    
class ResidualGroup(nn.Module):
    def __init__(self, InputChannels, BlockConstructors):
        super(ResidualGroup, self).__init__()
        
        self.Layers = nn.ModuleList([Block(**Arguments) for Block, Arguments in BlockConstructors])
        self.Alphas = nn.ParameterList([torch.nn.Parameter(torch.zeros(InputChannels)) for _ in range(len(self.Layers))])

    def forward(self, x):
        AccumulatedVariance = torch.ones([]).to(x.device)
        for Alpha, Layer in zip(self.Alphas, self.Layers):
            x = Layer(x, InputGain=torch.rsqrt(AccumulatedVariance) / math.sqrt(len(self.Layers)), ResidualGain=Alpha)
            AccumulatedVariance = AccumulatedVariance + Alpha * Alpha / len(self.Layers)
        
        return x, AccumulatedVariance
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        self.FastPath = InputChannels == OutputChannels
        
        if self.FastPath:
            self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        else:
            self.Resampler = InplaceUpsampler(ResamplingFilter)
            self.DuplicationRate = OutputChannels * 4 // InputChannels
        
    def forward(self, x, Gain):
        x = x * Gain.view(1, -1, 1, 1).to(x.dtype)
        
        if self.FastPath:
            return self.Resampler(x)
        else:
            return self.Resampler(x.repeat_interleave(self.DuplicationRate, dim=1))
        
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        self.FastPath = InputChannels == OutputChannels
        
        if self.FastPath:
            self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        else:
            self.Resampler = InplaceDownsampler(ResamplingFilter)
            self.ReductionRate = InputChannels * 4 // OutputChannels
        
    def forward(self, x, Gain):
        x = self.Resampler(x * Gain.view(1, -1, 1, 1).to(x.dtype))
        
        if self.FastPath:
            return x
        else:
            return x.view(x.shape[0], -1, self.ReductionRate, x.shape[2], x.shape[3]).mean(dim=2)
    
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
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, KernelSize=1)
        self.Gain = torch.nn.Parameter(torch.ones([]))
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = Linear(ConditionDimension, ConditionEmbeddingDimension)
        
    def forward(self, x, y=None):
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        x = self.BasisLayer(x).to(torch.bfloat16)
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x)

        return self.AggregationLayer(x, Gain=self.Gain * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead))
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.BasisLayer = DiscriminativeBasis(WidthPerStage[-1], 1 if ConditionDimension is None else ConditionEmbeddingDimension)
        self.ExtractionLayer = Convolution(3, WidthPerStage[0], KernelSize=1)
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = Linear(ConditionDimension, ConditionEmbeddingDimension)
            self.EmbeddingGain = 1 / math.sqrt(ConditionEmbeddingDimension)
        
    def forward(self, x, y=None):
        x = self.ExtractionLayer(x.to(torch.bfloat16))
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x)
        
        x = self.BasisLayer(x.to(torch.float32), Gain=torch.rsqrt(AccumulatedVariance))
        x = (x * self.EmbeddingLayer(y, Gain=self.EmbeddingGain)).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])
