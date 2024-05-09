import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class cosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(cosLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.scale = 0.09

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.000001)

        L_norm = torch.norm(self.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized, weight_normalized.transpose(0, 1))
        scores = cos_dist / self.scale
        return scores

class CustomViT(nn.Module):
    def __init__(self, n_classes):
        super(CustomViT, self).__init__()
        # Load the pretrained ViT model
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        num_ftrs = self.model.head.in_features

        # Replace the classifier head with a cosLinear layer for PCR-forward operation
        self.pcrLinear = cosLinear(num_ftrs, n_classes)
        
        # Override the original classification head to prevent automatic application
        self.model.head = nn.Identity()

    def forward_features(self, x):
        # Extract features from the transformer model
        features = self.model(x)
        return features

    def forward(self, x):
        # Compute features
        features = self.forward_features(x)
        # Compute logits using the cosLinear layer
        logits = self.pcrLinear(features)
        return logits

    def pcrForward(self, x):
        # Compute features
        features = self.forward_features(x)
        # Compute logits using the cosLinear layer
        pcr_logits = self.pcrLinear(features)
        return pcr_logits, features

def ViT_pretrained_with_pcr(n_classes):
    model = CustomViT(n_classes)
    return model



class SupConViT(nn.Module):
    """ViT backbone + projection head for n_classes"""
    def __init__(self, n_classes, head='mlp'):
        super(SupConViT, self).__init__()

        # Load the pretrained ViT model and remove its classification head
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder.head = nn.Identity()

        dim_in = self.encoder.num_features  # Get the feature dimension from ViT

        # Define the projection head
        if head == 'linear':
            self.head = nn.Linear(dim_in, n_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, n_classes)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        # Extract features using the ViT encoder
        feat = self.encoder(x)

        # Apply the projection head if it exists
        if self.head:
            feat = self.head(feat)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        # Directly use the ViT encoder for extracting features
        return self.encoder(x)
