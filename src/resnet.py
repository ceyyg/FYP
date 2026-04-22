import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ResNet18(nn.Module):
  """
  Implements ResNet18 with a modified head for binary classification with layer 4 unfrozen.
  """
  def __init__(self):
    super().__init__()
    # Load ImageNet pre-trained weights
    self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freezing the layers of ResNet
    for param in self.backbone.parameters():
      param.requires_grad = False

    # Unfreezing layer 4 to enable gradients computation
    for param in self.backbone.layer4.parameters():
      param.requires_grad = True

    # Replace FC layer for Gender classification (Female, Male)
    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

  def forward(self, x):
    return self.backbone(x)

model = ResNet18()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
del model





