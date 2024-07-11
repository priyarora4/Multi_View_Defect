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
from krevera_project_segment import NestedUNet


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
        # class_labels = torch.argmax(targets, dim=1)
        focal_loss = self.focal_loss(inputs, targets)
        return focal_loss



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

class MultiViewCNN(nn.Module):
    def __init__(self, nested_unet, num_classes):
        super().__init__()
        self.nested_unet = nested_unet
        self.features = None

        # Register a forward hook to capture the x0_4 output
        self._register_hook()
        self.conv_reduce = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # Second reduction step
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
        )
        self.squeeze_excite = Squeeze_Excite(32*5, 16)
        self.global_pool = nn.AdaptiveAvgPool2d(1) # Global pooling layer
        self.fc1 = nn.Linear(32, 32) # Fully connected layer
        self.fc2 = nn.Linear(32, num_classes) # Fully connected layer


    def _register_hook(self):
        # The hook will store the output of x0_4 in self.features
        def hook(module, input, output):
            self.features = output
        
        # Assuming the conv0_4 layer is the one we need (you need to confirm this from your model architecture)
        self.nested_unet.conv0_4.register_forward_hook(hook)

    def forward(self, x):
        # Forward pass through NestedUNet
        batch_size = x.shape[0]
        num_views = x.shape[1]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = self.nested_unet(x) # self.features now contains the output of x0_4 from nested unet model
        # note that x refers to the output segmentation prediction of the nested unet model

        features = self.conv_reduce(self.features)
        # we can concat the features maps of each view along the channel dimension
        features = features.view(batch_size, num_views*features.shape[1], features.shape[2], features.shape[3])
        # so now we have the features in the shape of [batch_size, num_views*C, H, W]
        # we can apply channel attention
        features = self.squeeze_excite(features)
        #fusion section. Add feature maps across views dimension. 
        features = features.view(batch_size, num_views, -1, features.shape[2], features.shape[3])
        features = torch.sum(features, dim=1)
        # now we have the fused features in the shape of [batch_size, C, H, W]
        features = self.global_pool(features)
        features = features.view(batch_size, -1)
        features = self.fc1(features)
        features = F.relu(features)
        features = self.fc2(features)

        return features



if __name__ == "__main__":
    # Setup #########################################################################################################################
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    cores = os.cpu_count()
    print(f"Total Cores: {cores}")
    ###################################################################################################################################

    config = {}
    config["backbone_pretrain_path"] = "NestedUnet_Segmentation_best.pth"
    config['num_classes'] = 20
    config['Wandb_project_name']="MultiView_unet"
    config["Wandb_log"] = True
    config['epochs']=100
    config['batch_size']=2
    config["mean"] = np.array([0.4016, 0.3994, 0.4520])
    config["std"] = np.array([0.2425, 0.2206, 0.1979])

    config['input_h']=544
    config['input_w']=960
    config['alphas'] = torch.from_numpy(np.load("alphas.npy")).float()

    config['loss']='FocalLoss'

    config['optimizer']='Adam'
    config['lr']=1e-3
    config['weight_decay']=1e-4
    config['momentum']=0.9

    config["warmup_epoch_steps"]=3
    config['scheduler']=''
    config['min_lr']=1e-6
    config['milestones']='1,2'
    config['gamma']=2/3
    # config['early_stopping']=-1
    config['num_workers']= cores
    config["save_model_path"]="MultiView_Unet_best.pth"

    # Backbone Unet++ model #########################################################################################################
    backbone_unet = NestedUNet(num_classes=3, in_channels=3, out_channels=3)
    backbone_unet.load_state_dict(torch.load(config["backbone_pretrain_path"]))
    multiview_model = MultiViewCNN(backbone_unet, num_classes=config['num_classes']).to(device)
    ###################################################################################################################################


    # Datasets, Dataloaders #####################################################################################################################
    test_dataset = KreveraSyntheticDataset("./krevera_synthetic_dataset_test", 
                                           zero_one_normalize=True, 
                                           is_train=False, 
                                           num_bins=config['num_classes']
                                           )
    train_dataset = KreveraSyntheticDataset("./krevera_synthetic_dataset_train", 
                                            zero_one_normalize=True, 
                                            is_train=True, 
                                            num_bins=config['num_classes'],
                                            # class_weights_file_path="./100_class_weights.npy",
                                            )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    val_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [val_size, train_size])
    val_dataset.dataset.is_train = False
    val_dataset.dataset.set_transforms()
    # create small dataset of size 2
    # this will be used to test training loop
    # notice there are no transforms applied to this dataset
    dummy_dataset = torch.utils.data.Subset(val_dataset, list(range(2)))

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=config['batch_size'], 
                                               shuffle=True, 
                                               drop_last=True, 
                                               num_workers=config['num_workers']
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=config['batch_size'], 
                                             shuffle=False,  
                                             drop_last=True, 
                                             num_workers=config['num_workers'])
    ###################################################################################################################################


    # Loss, Optimizer, Scheduler #####################################################################################################
    optimizer = torch.optim.Adam(multiview_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = FocalLoss(alpha=config['alphas'], gamma=2.0).to(device)
    # criterion = nn.CrossEntropyLoss()
    def lambda1(current_epoch):
        if current_epoch < config["warmup_epoch_steps"]:
            return 0.001 + (1.0 - 0.001) * (current_epoch / config["warmup_epoch_steps"])
        return 1.0
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    ###################################################################################################################################


    ### wandb logging ############################################################################################################
    if config["Wandb_log"]:
        import wandb
        wandb.init(
        # set the wandb project where this run will be logged
        project="multi-view-classification-basic",

        # track hyperparameters and run metadata
        config=config
        )
    ###################################################################################################################################

        ### Training Loop ###
    torch.cuda.empty_cache()
    best_val_accuracy = 0.0
    multiview_model.to(device)
    scaler = GradScaler()
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}")
        # train one epoch
        multiview_model.train()
        train_loss = 0.0
        train_corrects = 0

        for batch in train_loader:

            input_rgbs = batch["input_rgb"].to(device)
            bin_nums = batch["bin_num"].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = multiview_model(input_rgbs)
                loss = criterion(outputs, bin_nums)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            corrects = (torch.argmax(outputs, 1) == bin_nums).sum().item()
            train_corrects += corrects
            if config["Wandb_log"]:
                wandb.log({"batch_train_loss": loss.item(), "batch_train_accuracy": corrects/config['batch_size']})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})


        train_loss /= len(train_loader)
        train_accuracy = train_corrects / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        if config["Wandb_log"]:
            wandb.log({"epoch_train_loss": train_loss, "epoch_train_accuracy": train_accuracy})
        
        ### eval the model
        multiview_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_corrects = 0
            with autocast():
                for batch in val_loader:
                    input_rgbs = batch["input_rgb"].to(device)
                    bin_nums = batch["bin_num"].to(device)
                    outputs = multiview_model(input_rgbs)
                    loss = criterion(outputs, bin_nums)
                    val_loss += loss.item()
                    val_corrects += (torch.argmax(outputs, 1) == bin_nums).sum().item()
                val_loss /= len(val_loader)
                val_accuracy = val_corrects / len(val_loader.dataset)
                best_val_accuracy = max(best_val_accuracy, val_accuracy)
                if val_accuracy == best_val_accuracy:
                    torch.save(multiview_model.state_dict(), "best_model_weighted.pth")
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                if config["Wandb_log"]:
                    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
        if epoch < config["warmup_epoch_steps"]:
            scheduler_warmup.step()