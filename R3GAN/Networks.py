import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler, InplaceUpsampler, InplaceDownsampler
from .MagnitudePreservingLayers import LeakyReLU, Convolution, Linear, BiasedPointwiseConvolutionWithModulation, CosineAttention, GenerativeBasis, DiscriminativeBasis, BoundedParameter, ClassEmbedder

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, EmbeddingDimension, ChannelsPerHead):
        super(MultiHeadSelfAttention, self).__init__()

        self.QKVLayer = BiasedPointwiseConvolutionWithModulation(InputChannels, HiddenChannels * 3, EmbeddingDimension, Centered=True)
        self.ProjectionLayer = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.Heads = HiddenChannels // ChannelsPerHead

    def forward(self, x, w, InputGain, ResidualGain):
        QKVLayer = lambda y: self.QKVLayer(y, w, Gain=InputGain.view(1, -1, 1, 1))
        ProjectionLayer = lambda y: self.ProjectionLayer(y, Gain=ResidualGain.view(-1, 1, 1, 1))
        
        return x + CosineAttention(x, self.Heads, QKVLayer, ProjectionLayer)

class FeedForwardNetwork(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, EmbeddingDimension, ChannelsPerGroup, KernelSize):
        super(FeedForwardNetwork, self).__init__()
        
        self.LinearLayer1 = BiasedPointwiseConvolutionWithModulation(InputChannels, HiddenChannels, EmbeddingDimension, Centered=True)
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, Centered=True)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.NonLinearity = LeakyReLU()
        
    def forward(self, x, w, InputGain, ResidualGain):
        y = self.LinearLayer1(x, w, Gain=InputGain.view(1, -1, 1, 1))
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y), Gain=ResidualGain.view(-1, 1, 1, 1))
        
        return x + y
    
class ResidualGroup(nn.Module):
    def __init__(self, InputChannels, BlockConstructors):
        super(ResidualGroup, self).__init__()
        
        self.Layers = nn.ModuleList([Block(**Arguments) for Block, Arguments in BlockConstructors])
        self.ParametrizedAlphas = nn.ModuleList([BoundedParameter(InputChannels) for _ in range(len(self.Layers))])

    def forward(self, x, w):
        AccumulatedVariance = torch.ones([]).to(x.device)
        for ParametrizedAlpha, Layer in zip(self.ParametrizedAlphas, self.Layers):
            Alpha = ParametrizedAlpha()
            x = Layer(x, w, InputGain=torch.rsqrt(AccumulatedVariance), ResidualGain=Alpha)
            AccumulatedVariance = AccumulatedVariance + Alpha * Alpha
        
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
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, InputDimension, OutputDimension, HiddenDimension, ActivateOutput):
        super(MultiLayerPerceptron, self).__init__()
        
        self.ActivateOutput = ActivateOutput
        
        self.LinearLayer1 = Linear(InputDimension, HiddenDimension, Centered=True)
        self.LinearLayer2 = Linear(HiddenDimension, OutputDimension, Centered=True)
        self.NonLinearity = LeakyReLU()
        
    def forward(self, x):
        x = self.LinearLayer2(self.NonLinearity(self.LinearLayer1(x)))
        
        return self.NonLinearity(x) if self.ActivateOutput else x
    
class GenerativeHead(nn.Module):
    def __init__(self, InputDimension, OutputChannels, HiddenDimension):
        super(GenerativeHead, self).__init__()
        
        self.Basis = GenerativeBasis(OutputChannels)
        self.FullyConnectedLayer = MultiLayerPerceptron(InputDimension, OutputChannels, HiddenDimension, False)
        
    def forward(self, x):
        return self.Basis(self.FullyConnectedLayer(x))
    
class DiscriminativeHead(nn.Module):
    def __init__(self, InputChannels, OutputDimension, HiddenDimension):
        super(DiscriminativeHead, self).__init__()
        
        self.Basis = DiscriminativeBasis(InputChannels)
        self.FullyConnectedLayer = MultiLayerPerceptron(InputChannels, OutputDimension, HiddenDimension, False)
        
    def forward(self, x, Gain):
        return self.FullyConnectedLayer(self.Basis(x, Gain=Gain.view(-1, 1, 1, 1)))
    
def BuildResidualGroups(WidthPerStage, BlocksPerStage, EmbeddingDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead):
    ResidualGroups = []
    for Width, Blocks in zip(WidthPerStage, BlocksPerStage):
        BlockConstructors = []
        for BlockType in Blocks:
            if BlockType == 'FFN':
                BlockConstructors += [(FeedForwardNetwork, dict(InputChannels=Width, HiddenChannels=round(Width * FFNWidthRatio), EmbeddingDimension=EmbeddingDimension, ChannelsPerGroup=ChannelsPerConvolutionGroup, KernelSize=KernelSize))]
            elif BlockType == 'Attention':
                BlockConstructors += [(MultiHeadSelfAttention, dict(InputChannels=Width, HiddenChannels=round(Width * AttentionWidthRatio), EmbeddingDimension=EmbeddingDimension, ChannelsPerHead=ChannelsPerAttentionHead))]
        ResidualGroups += [ResidualGroup(Width, BlockConstructors)]
    return ResidualGroups
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, ModulationDimension, WidthPerStage, BlocksPerStage, MLPWidthRatio, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, ModulationDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead))
        self.TransitionLayers = nn.ModuleList([UpsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.Head = GenerativeHead(NoiseDimension + ClassEmbeddingDimension, WidthPerStage[0], WidthPerStage[0] * MLPWidthRatio)
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, KernelSize=1)
        self.Gain = nn.Parameter(torch.ones([]))
        
        self.MappingLayer = MultiLayerPerceptron(NoiseDimension + ClassEmbeddingDimension, ModulationDimension, ModulationDimension * MLPWidthRatio, True)
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension)
        
    def forward(self, x, y=None):
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        w = self.MappingLayer(x)
        x = self.Head(x).to(torch.bfloat16)
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x, w)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x, w)

        return self.AggregationLayer(x, Gain=self.Gain * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, ModulationDimension, WidthPerStage, BlocksPerStage, MLPWidthRatio, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, ModulationDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead))
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.Head = DiscriminativeHead(WidthPerStage[-1], 1 if NumberOfClasses is None else ClassEmbeddingDimension, WidthPerStage[-1] * MLPWidthRatio)
        self.ExtractionLayer = Convolution(3, WidthPerStage[0], KernelSize=1)
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension)
            self.MappingLayer = MultiLayerPerceptron(ClassEmbeddingDimension, ModulationDimension, ModulationDimension * MLPWidthRatio, True)
        
    def forward(self, x, y=None):
        if hasattr(self, 'EmbeddingLayer'):
            y = self.EmbeddingLayer(y)
            w = self.MappingLayer(y)
        else:
            w = None
        x = self.ExtractionLayer(x.to(torch.bfloat16))
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x, w)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x, w)
        
        x = self.Head(x.to(torch.float32), Gain=torch.rsqrt(AccumulatedVariance))
        x = (x * y / math.sqrt(y.shape[1])).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])
