# testing loss
import torch
import torch.nn as nn

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=None, smooth=1e-5):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = weights  # Expected to be a tensor of shape [C], where C is the number of classes

    def forward(self, inputs, targets):
        # Assuming inputs are raw logits and targets are indices of the ground truth classes
        inputs = torch.softmax(inputs, dim=1)
        
        # Create one-hot encoded targets
        # targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        # Calculate per-class Dice coefficient
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        cardinality = torch.sum(inputs + targets, dim=(2, 3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Apply weights
        if self.weights is not None:
            dice_loss = dice_loss * self.weights.to(dice_loss.device)

        # Instead of mean, you can now do a weighted sum based on the weights provided
        # loss_per_batch = dice_loss.sum(dim=1)
        total_loss = dice_loss.sum() / self.weights.sum()
        return total_loss

class CombinedCrossEntropyDiceLoss(nn.Module):
    def __init__(self, dice_weights=None, dice_weight=1.0, ce_weight=0.5):
        super(CombinedCrossEntropyDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = WeightedDiceLoss(weights=dice_weights)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return combined_loss

# Create a dummy model output and target
dummy_output = torch.randn(8, 3, 544, 960)
dummy_target = torch.randn(8, 3, 544, 960)
criterion = CombinedCrossEntropyDiceLoss(dice_weights=torch.tensor([0.1, 0.2, 0.7]))

loss = criterion(dummy_output, dummy_target)

