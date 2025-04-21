import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ResNetEmbedder(nn.Module):
    def __init__(self, resnet_type='resnet34', pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet
        resnet = getattr(models, resnet_type)(pretrained=pretrained)

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove fc layer

        # Output feature dimension
        self.embedding_dim = resnet.fc.in_features

    def forward(self, x):
        """
        x: (B, C, H, W) image tensor
        returns: (B, D) embedding tensor
        """
        with torch.no_grad():  # if you want inference only
            features = self.feature_extractor(x)  # (B, D, 1, 1)
            features = features.view(features.size(0), -1)  # (B, D)
        return features

