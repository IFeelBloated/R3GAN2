import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler, InplaceUpsampler, InplaceDownsampler
from .MagnitudePreservingLayers import LeakyReLU, Convolution, Linear, BiasedPointwiseConvolutionWithModulation, NoisyBiasedPointwiseConvolutionWithModulation, CosineAttention, GenerativeBasis, DiscriminativeBasis, BoundedParameter, ClassEmbedder
from .RoPE import RotaryPositionEmbedding

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, EmbeddingDimension, ChannelsPerHead):
        super(MultiHeadSelfAttention, self).__init__()

        self.QKVLayer = BiasedPointwiseConvolutionWithModulation(InputChannels, HiddenChannels * 3, EmbeddingDimension, Centered=True)
        self.ProjectionLayer = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.Heads = HiddenChannels // ChannelsPerHead
        self.Sinks = nn.Parameter(torch.zeros(self.Heads))
        self.RoPE = RotaryPositionEmbedding(embed_dim=HiddenChannels, num_heads=self.Heads)

    def forward(self, x, w, InputGain, ResidualGain):
        QKVLayer = lambda y: self.QKVLayer(y, w, Gain=InputGain.view(1, -1, 1, 1))
        ProjectionLayer = lambda y: self.ProjectionLayer(y, Gain=ResidualGain.view(-1, 1, 1, 1))
        
        N, C, H, W = x.shape
        RoPE = self.RoPE(H=H, W=W, device=x.device)
        
        return x + CosineAttention(x, self.Heads, QKVLayer, ProjectionLayer, self.Sinks, RoPE, a=8)

class FeedForwardNetwork(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, EmbeddingDimension, ChannelsPerGroup, KernelSize, Noise):
        super(FeedForwardNetwork, self).__init__()
        
        if Noise:
            self.LinearLayer1 = NoisyBiasedPointwiseConvolutionWithModulation(InputChannels, HiddenChannels, EmbeddingDimension, Centered=True)
        else:
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
    

class GenerativeHead(nn.Module):
    def __init__(self, InputDimension, OutputChannels, ResamplingFilter):
        super(GenerativeHead, self).__init__()

        self.LinearLayer = Convolution(InputDimension, OutputChannels * 4, KernelSize=1, Centered=True)


        BlockConstructors1 = [(FeedForwardNetwork, dict(InputChannels=OutputChannels, 
            HiddenChannels=round(OutputChannels * 2), 
            EmbeddingDimension=None, 
            ChannelsPerGroup=32, 
            KernelSize=3, Noise=True))]

        self.Res1 = ResidualGroup(OutputChannels, BlockConstructors1)
        self.Trans1 = UpsampleLayer(OutputChannels, OutputChannels, ResamplingFilter)

        BlockConstructors2 = [(FeedForwardNetwork, dict(InputChannels=OutputChannels, 
            HiddenChannels=round(OutputChannels * 2), 
            EmbeddingDimension=None, 
            ChannelsPerGroup=32, 
            KernelSize=3, Noise=True))]

        self.Res2 = ResidualGroup(OutputChannels, BlockConstructors2)
        self.Trans2 = UpsampleLayer(OutputChannels, OutputChannels, ResamplingFilter)
        
    def forward(self, x):
        x = nn.functional.pixel_shuffle(self.LinearLayer(x.view(x.shape[0], -1, 1, 1)), 2)
        x, AccumulatedVariance = self.Res1(x, None)
        x = self.Trans1(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.Res2(x, None)
        x = self.Trans2(x, Gain=torch.rsqrt(AccumulatedVariance))

        return x

class DiscriminativeHead(nn.Module):
    def __init__(self, InputChannels, OutputDimension, ResamplingFilter):
        super(DiscriminativeHead, self).__init__()

        self.LinearLayer = Convolution(InputChannels * 4, OutputDimension, KernelSize=1, Centered=True)

        BlockConstructors1 = [(FeedForwardNetwork, dict(InputChannels=InputChannels, 
            HiddenChannels=round(InputChannels * 2), 
            EmbeddingDimension=None, 
            ChannelsPerGroup=32, 
            KernelSize=3, Noise=False))]

        self.trans1 = DownsampleLayer(InputChannels, InputChannels, ResamplingFilter)
        self.Res1 = ResidualGroup(InputChannels, BlockConstructors1)

        BlockConstructors2 = [(FeedForwardNetwork, dict(InputChannels=InputChannels, 
            HiddenChannels=round(InputChannels * 2), 
            EmbeddingDimension=None, 
            ChannelsPerGroup=32, 
            KernelSize=3, Noise=False))]

        self.trans2 = DownsampleLayer(InputChannels, InputChannels, ResamplingFilter)
        self.Res2 = ResidualGroup(InputChannels, BlockConstructors2)

        
    def forward(self, x, Gain):
        x = self.trans1(x, Gain=Gain)
        x, AccumulatedVariance = self.Res1(x, None)
        x = self.trans2(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.Res2(x, None)
        x = x * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1)
        x = self.LinearLayer(nn.functional.pixel_unshuffle(x, 2)).view(x.shape[0], -1)

        return x







def BuildResidualGroups(WidthPerStage, BlocksPerStage, EmbeddingDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead, Noise):
    ResidualGroups = []
    for Width, Blocks in zip(WidthPerStage, BlocksPerStage):
        BlockConstructors = []
        for BlockType in Blocks:
            if BlockType == 'FFN':
                BlockConstructors += [(FeedForwardNetwork, dict(InputChannels=Width, HiddenChannels=round(Width * FFNWidthRatio), EmbeddingDimension=EmbeddingDimension, ChannelsPerGroup=ChannelsPerConvolutionGroup, KernelSize=KernelSize, Noise=Noise))]
            elif BlockType == 'Attention':
                BlockConstructors += [(MultiHeadSelfAttention, dict(InputChannels=Width, HiddenChannels=round(Width * AttentionWidthRatio), EmbeddingDimension=EmbeddingDimension, ChannelsPerHead=ChannelsPerAttentionHead))]
        ResidualGroups += [ResidualGroup(Width, BlockConstructors)]
    return ResidualGroups
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, ModulationDimension, OutputChannels, WidthPerStage, BlocksPerStage, MLPWidthRatio, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        ModulationDimension = None
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, ModulationDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead, Noise=True))
        self.TransitionLayers = nn.ModuleList([UpsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.Head = GenerativeHead(NoiseDimension + ClassEmbeddingDimension, WidthPerStage[0], ResamplingFilter)
        self.AggregationLayer = Convolution(WidthPerStage[-1], OutputChannels, KernelSize=3, Centered=True)
        self.Gain = nn.Parameter(torch.ones(OutputChannels))
        self.Bias = nn.Parameter(torch.zeros(OutputChannels))
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension)
        
    def forward(self, x, y=None):
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        w = None
        x = self.Head(x).to(torch.bfloat16)
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x, w)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x, w)

        return self.AggregationLayer(x, Gain=self.Gain.view(-1, 1, 1, 1) * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1)) + self.Bias.view(1, -1, 1, 1)

class Discriminator(nn.Module):
    def __init__(self, ModulationDimension, InputChannels, WidthPerStage, BlocksPerStage, MLPWidthRatio, FFNWidthRatio, ChannelsPerConvolutionGroup, AttentionWidthRatio, ChannelsPerAttentionHead, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        ModulationDimension = None
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, ModulationDimension, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, AttentionWidthRatio, ChannelsPerAttentionHead, Noise=False))
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.Head = DiscriminativeHead(WidthPerStage[-1], 1 if NumberOfClasses is None else ClassEmbeddingDimension, ResamplingFilter)
        self.ExtractionLayer = Convolution(InputChannels, WidthPerStage[0], KernelSize=3, Centered=True)
        self.Gain = nn.Parameter(torch.ones(InputChannels))
        self.Bias = nn.Parameter(torch.zeros(InputChannels))
                
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension)
        
    def forward(self, x, y=None):
        if hasattr(self, 'EmbeddingLayer'):
            y = self.EmbeddingLayer(y)
        w = None
        x = self.ExtractionLayer((x + self.Bias.view(1, -1, 1, 1)).to(torch.bfloat16), Gain=self.Gain.view(1, -1, 1, 1))
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x, w)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x, w)
        
        x = self.Head(x.to(torch.float32), Gain=torch.rsqrt(AccumulatedVariance))
        x = (x * y / math.sqrt(y.shape[1])).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])
