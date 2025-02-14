import torch
import torch.nn as nn
import copy
import R3GAN.Networks

class Generator(nn.Module):
    def __init__(self, *args, **kw):
        super(Generator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Generator(*args, **config)
        self.z_dim = kw['NoiseDimension']
        self.c_dim = kw['c_dim']
        self.img_resolution = kw['img_resolution']
        
        self.Model.DataTypePerStage = [torch.bfloat16 for _ in self.Model.DataTypePerStage]
        self.Model.DataTypePerStage[0] = torch.float32
        
    def forward(self, x, c):
        return self.Model(x, c)
    
class Discriminator(nn.Module):
    def __init__(self, *args, **kw):
        super(Discriminator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Discriminator(*args, **config)
        
        self.Model.DataTypePerStage = [torch.bfloat16 for _ in self.Model.DataTypePerStage]
        self.Model.DataTypePerStage[-1] = torch.float32
        
    def forward(self, x, c):
        return self.Model(x, c)