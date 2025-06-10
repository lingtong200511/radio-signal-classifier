import torch.nn as nn
import torchvision.models as models

class S2MResNet(nn.Module):
    def __init__(self, num_classes=11):
        super(S2MResNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
