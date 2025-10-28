"""
VGG16-based classifier for MFCC inputs.
We adapt torchvision VGG16 to accept 1-channel inputs and configurable output classes.
"""

import os

# Add src directory to path for imports
import sys

import torch
import torch.nn as nn
from torchvision import models

from .base import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.constants import DEFAULT_VGG_DROPOUT, DEFAULT_VGG_NUM_CLASSES, DEFAULT_VGG_PRETRAINED


class VGG16Classifier(BaseModel):
    def __init__(
        self,
        num_classes: int = DEFAULT_VGG_NUM_CLASSES,
        pretrained: bool = DEFAULT_VGG_PRETRAINED,
        dropout: float = DEFAULT_VGG_DROPOUT,
        num_mfcc_features: int = 13,
    ):
        super().__init__(model_name="VGG16")
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_mfcc_features = num_mfcc_features

        # Load VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)

        # Adapt first conv to 1 input channel (MFCC as single-channel image)
        old_conv1 = vgg.features[0]
        new_conv1 = nn.Conv2d(
            1,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )
        if pretrained and old_conv1.weight.shape[1] == 3:
            # Average RGB weights into 1 channel
            with torch.no_grad():
                new_conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)
                if old_conv1.bias is not None:
                    new_conv1.bias[:] = old_conv1.bias
        vgg.features[0] = new_conv1

        # Make pooling asymmetric to avoid collapsing narrow MFCC width
        # Replace 2x2 pools with (2,1) (downsample time, preserve freq width)
        new_features = []
        for m in vgg.features:
            if isinstance(m, nn.MaxPool2d):
                new_features.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0))
            else:
                new_features.append(m)
        vgg.features = nn.Sequential(*new_features)

        # Remove VGG's default avgpool/classifier; we'll add a fixed pool over time only
        # After 5 pools with (2,1), input time=1292 becomes 40; width stays 13.
        # Use a fixed AvgPool2d to collapse time to 1 so ONNX has fixed kernel sizes.
        vgg.avgpool = nn.Identity()
        self.time_pool = nn.AvgPool2d(kernel_size=(40, 1), stride=(1, 1))

        # Lightweight custom head for 512 x 1 x num_mfcc_features maps
        # After VGG features and time pooling, we get (batch, 512, 1, num_mfcc_features)
        # After flattening: (batch, 512 * num_mfcc_features)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * num_mfcc_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, num_classes),
        )

        # Optional dropout tweak
        if isinstance(vgg.classifier[5], nn.Dropout):
            vgg.classifier[5].p = self.dropout

        self.vgg = vgg

        # Store config
        self.model_config = {
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input MFCC shapes:
        # - (batch, time, features) -> reshape to (batch, 1, time, features)
        # - (batch, 1, H, W) already image-like
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            # If given (batch, H, W) format, add channel
            if x.shape[1] != 1 and x.shape[1] != 3:
                x = x.unsqueeze(1)
        feats = self.vgg.features(x)
        feats = self.time_pool(feats)
        out = self.head(feats)
        return torch.log_softmax(out, dim=1)
