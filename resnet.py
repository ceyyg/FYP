import torch
import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Mdoule):
    def __init__(self):
        super (Resnet, self).__init__()
        self.backbone = models.resnet18(pretrained = True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.linear(num_features, 2)

    def forward(self, x):
        return self.backbone(x)
    
    
