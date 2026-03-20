import torch
import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self):
        super (Resnet, self).__init__()
        self.backbone = models.resnet18(pretrained = True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.backbone(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= Resnet().to(device) 

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

