import torch
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import krevera_synthetic_dataset
import json
from tqdm import tqdm
import torch.nn as nn

import importlib
importlib.reload(krevera_synthetic_dataset)
from krevera_synthetic_dataset import KreveraSyntheticDataset
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F




# model ###########################################################################################################################

class Squeeze_Excite(nn.Module):
    
    def __init__(self,channel,reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class VGGBlock(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excite(out_channels,8)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.SE(out)
        
        return(out)

class NestedUNet(nn.Module):
    
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, is_backbone=False, **kwargs):
        super().__init__()

        self.is_backbone = is_backbone
        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # if self.is_backbone:
        #     return x0_4
        # else:
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

###################################################################################################################################

# loss, evaluation metrics ###########################################################################################################################

# class BCEDiceLoss(nn.Module):
    
#     def __init__(self):
#         super().__init__()
    
#     def forward(self,input,target):
#         bce = F.binary_cross_entropy_with_logits(input,target)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num,-1)
#         target = target.view(num,-1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         return (0.5 * bce + dice)

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
        dice_loss = dice_loss.sum() / inputs.shape[0]
        return dice_loss

class CombinedCrossEntropyDiceLoss(nn.Module):
    def __init__(self, dice_weights=None, dice_weight=0.9, ce_weight=0.1):
        super(CombinedCrossEntropyDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = WeightedDiceLoss(weights=dice_weights)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        # print("ce loss and dice loss", ce_loss, dice_loss)
        combined_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return combined_loss


def dice_score(preds, targets, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)   
    preds_one_hot = torch.zeros_like(preds).scatter_(1, preds.argmax(1, keepdim=True), 1)  
    intersection = torch.sum(preds_one_hot * targets, dim=(2, 3))
    cardinality = torch.sum(preds_one_hot + targets, dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (cardinality + smooth)
    # Average dice score across batch
    return dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.focal_loss = torch.hub.load(
                            'adeelh/pytorch-multi-class-focal-loss',
                            model='FocalLoss',
                            alpha=alpha.to(device) if alpha is not None else None,
                            gamma=2,
                            reduction='mean',
                            force_reload=False, 
                        )

    def forward(self, inputs, targets):
        class_labels = torch.argmax(targets, dim=1)
        focal_loss = self.focal_loss(inputs, class_labels)
        return focal_loss

class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, dice_weights=None, dice_weight=0.1, focal_weight=1.0, alpha=None, gamma=2):
        super(CombinedFocalDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = WeightedDiceLoss(weights=dice_weights)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        combined_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return combined_loss

def tensor_to_img(tensor):
    return tensor.cpu().numpy()


###################################################################################################################################








if __name__ == "__main__":
    
    # Setup #########################################################################################################################
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    cores = os.cpu_count()
    print(f"Total Cores: {cores}")
    ###################################################################################################################################

    # Config #########################################################################################################################

    config = {}
    config['Wandb_project_name']="Segmentation_krevera"
    config["Wandb_log"] = True
    config['epochs']=100
    config['batch_size']=2
    config["mean"] = np.array([0.4016, 0.3994, 0.4520])
    config["std"] = np.array([0.2425, 0.2206, 0.1979])
    config['dice_weights'] = torch.tensor([0.1, 0.1, 0.8])  

    config['arch']='NestedUNet'
    config['deep_supervision']=False
    config['input_channels']=3
    config['num_classes']=3
    config['input_h']=544
    config['input_w']=960

    config['loss']='BCEDiceLoss'

    config['optimizer']='Adam'
    config['lr']=1e-3
    config['weight_decay']=1e-4
    config['momentum']=0.9
    config['nesterov']=False

    config["warmup_epoch_steps"]=1
    config['scheduler']=''
    config['min_lr']=1e-6
    config['factor']=0.1
    config['patience']=10
    config['milestones']='1,2'
    config['gamma']=2/3
    # config['early_stopping']=-1
    config['num_workers']= cores
    config["save_model_path"]="Nested_unet_dummy_best.pth"

    ###################################################################################################################################


    # wandb login ####################################################################################################################
    if config["Wandb_log"]:
        import wandb
        wandb.init(
        project=config['Wandb_project_name'],
        config=config.copy()
        )
    ###################################################################################################################################


    
    # dataset #########################################################################################################################
    test_dataset = KreveraSyntheticDataset("./krevera_synthetic_dataset_test", 
                                        zero_one_normalize=True, 
                                        is_train=False
                                        )
    train_dataset = KreveraSyntheticDataset("./krevera_synthetic_dataset_train", 
                                            zero_one_normalize=True, 
                                            is_train=True, 
                                            )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    val_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [val_size, train_size])
    val_dataset.is_train = False
    # create small dataset of size 2
    # this will be used to test training loop
    # notice there are no transforms applied to this dataset
    dummy_dataset = torch.utils.data.Subset(test_dataset, list(range(87,88)))


    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Val Dataset Length: {len(val_dataset)}")
    print(f"Test Dataset Length: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=config['num_workers'])
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,  drop_last=True, num_workers=config['num_workers'])

    #dummy loaders
    # train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=config['batch_size'], shuffle=False,  drop_last=True, num_workers=2)
    # val_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=config['batch_size'], shuffle=False,  drop_last=True, num_workers=2)
    ###################################################################################################################################



    # Initalize Model, optimzer, loss, and scheduler ##################################################################################
    model = NestedUNet(num_classes=config['num_classes'], input_channels=config['input_channels'], deep_supervision=config['deep_supervision'])
    model.to(device)
    alpha = torch.tensor([0.01, 0.01, 0.98])
    criterion = CombinedFocalDiceLoss(dice_weights=config['dice_weights'], alpha=alpha, gamma=2)


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    warmup_epochs = config["warmup_epoch_steps"]
    lr_warmup_factor = 1e-3
    def lambda1(current_epoch):
        if current_epoch < warmup_epochs:
            return lr_warmup_factor + (1.0 - lr_warmup_factor) * (current_epoch / warmup_epochs)
        return 1.0
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=config['patience'], verbose=True, min_lr=config['min_lr'])

    ###################################################################################################################################





    # Training Loop ###################################################################################################################

    torch.cuda.empty_cache()
    best_val_dice = 0.0
    scaler = GradScaler()

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}")
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        i = -1 # for wandb logging image
        for batch in train_loader:
            i+=1 # for wandb logging image

            input_rgbs = batch["input_rgb"].to(device)
            input_rgbs = input_rgbs.view(input_rgbs.shape[0]*input_rgbs.shape[1],input_rgbs.shape[2],input_rgbs.shape[3],input_rgbs.shape[4])
            target_masks = batch["target_segmentation"].to(device)
            target_masks = target_masks.view(target_masks.shape[0]*target_masks.shape[1],target_masks.shape[2],target_masks.shape[3],target_masks.shape[4])
            optimizer.zero_grad()
            with autocast():
                outputs = model(input_rgbs)
                if config['deep_supervision']:
                    loss_value = sum([criterion(output, target_masks) for output in outputs])
                    loss_value /= len(outputs)
                else:
                    loss_value = criterion(outputs, target_masks)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss_value.item()
            if config['deep_supervision']:
                dice = dice_score(outputs[-1], target_masks).item()
            else:
                dice = dice_score(outputs, target_masks).item()

            train_dice += dice
            if config["Wandb_log"]:
                wandb.log({"batch_train_loss": loss_value.item(), "batch_avg_dice": dice/config['batch_size']})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
                if i == 0:  # Log only the first batch's images
                    # Convert the first image in the batch to log
                    if config['deep_supervision']:
                        prediction = outputs[-1]
                    else:
                        prediction = outputs

                    logged_image = np.transpose(tensor_to_img(input_rgbs[0]), (1, 2, 0))*config["std"] + config["mean"]
                    logged_mask = np.transpose(tensor_to_img(target_masks[0]), (1, 2, 0))
                    preds = torch.softmax(prediction, dim=1)   
                    preds_one_hot = torch.zeros_like(preds).scatter_(1, preds.argmax(1, keepdim=True), 1)[0] 
                    logged_prediction = np.transpose(tensor_to_img(preds_one_hot), (1, 2, 0))
                    
                    # Log images to wandb
                    wandb.log({
                        "examples": [
                            wandb.Image(logged_image, caption="Input Image"),
                            wandb.Image(logged_mask, caption="Ground Truth Mask"),
                            wandb.Image(logged_prediction, caption="Predicted Mask")
                        ]
                    })
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        if config["Wandb_log"]:
            wandb.log({"epoch_train_loss": train_loss, "epoch_train_dice": train_dice})
        
        # evaluate the model on validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_dice = 0.0
            for batch in val_loader:
                input_rgbs = batch["input_rgb"].to(device)
                input_rgbs = input_rgbs.view(input_rgbs.shape[0]*input_rgbs.shape[1],input_rgbs.shape[2],input_rgbs.shape[3],input_rgbs.shape[4])
                target_masks = batch["target_segmentation"].to(device)
                target_masks = target_masks.view(target_masks.shape[0]*target_masks.shape[1],target_masks.shape[2],target_masks.shape[3],target_masks.shape[4])
                outputs = model(input_rgbs)
                if config['deep_supervision']:
                    loss_value = sum([criterion(output, target_masks) for output in outputs])
                    loss_value /= len(outputs)
                else:
                    loss_value = criterion(outputs, target_masks)
                val_loss += loss_value.item()
                if config['deep_supervision']:
                    dice = dice_score(outputs[-1], target_masks).item()
                else:
                    dice = dice_score(outputs, target_masks).item()
                val_dice += dice

            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
            best_val_dice = max(best_val_dice, val_dice)
            if val_dice == best_val_dice:
                torch.save(model.state_dict(), config["save_model_path"])
            
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            if config["Wandb_log"]:
                wandb.log({"val_loss": val_loss, "val_dice": val_dice})

        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler_reduce.step(val_dice)
            # scheduler_reduce.step(val_dice)

    print("Training Complete")