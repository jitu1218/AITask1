"""Handwritten digit classifier. Inputs are N x 1 x 8 x 8, in [0, 1]."""
from torch import nn


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for inputs, outputs in [(1, 24), (24, 48), (48, 64)]:
            blocks.extend([
                nn.Conv2d(inputs, outputs, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 10))

    def forward(self, images):
        return self.classifier(self.features(images))
