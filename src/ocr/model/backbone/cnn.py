import torch
import torch.nn as nn

import ocr.model.backbone.vgg as vgg
import ocr.model.backbone.resnet as resnet


class CNN(nn.Module):
    def __init__(self, backbone, **kwargs) -> None:
        super(CNN, self).__init__()
        
        if backbone == "vgg11_bn":
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == "vgg19_bn":
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == "resnet50":
            self.model = resnet.Resnet50(**kwargs)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv1x1':
                param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True