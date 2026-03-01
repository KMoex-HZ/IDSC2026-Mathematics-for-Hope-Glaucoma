import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class GlaucomaEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(GlaucomaEfficientNet, self).__init__()
        # Call the pre-trained architecture optimized using Compound Scaling
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Extract the number of features from the final layer before classification
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the final classification layer with a Fully Connected Layer for binary classification (1 output)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# Custom Loss Function to handle class imbalance and incorporate Quality Scores
class WeightedQualityBCE(nn.Module):
    def __init__(self, pos_weight=None):
        super(WeightedQualityBCE, self).__init__()
        # Set pos_weight to handle class imbalance (e.g., 0.363 based on GON-/GON+ ratio)
        # Prevents model bias and improves recognition of the minority class
        self.bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        
    def forward(self, logits, targets, quality_weights):
        # Calculate the base loss between predictions and ground truth labels
        base_loss = self.bce_with_logits(logits, targets)
        
        # Weight the loss based on image Quality Score
        # High-quality images with incorrect predictions will result in an amplified penalty
        weighted_loss = base_loss * quality_weights
        
        return weighted_loss.mean()