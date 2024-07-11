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

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



# Load in datasets
test_dataset = KreveraSyntheticDataset("./krevera_synthetic_dataset_test", zero_one_normalize=True, is_train=False)
train_dataset = KreveraSyntheticDataset("./krevera_synthetic_dataset_train", 
                                        zero_one_normalize=True, 
                                        is_train=True, 
                                        class_weights_file_path="./100_class_weights.npy")
class_weights = torch.tensor(train_dataset.class_weights).to(device)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
val_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [val_size, train_size])

# create small dataset of size 2
# this will be used to test training loop
# notice there are no transforms applied to this dataset
dummy_dataset = torch.utils.data.Subset(train_dataset, list(range(2)))


##### visualize data ######################################################################################################
# datapoint = dummy_dataset[0]

# # # fig, axs = plt.subplots(2, 5, figsize=(15, 6))

# # fig, axs = plt.subplots(5, 2, figsize=(20, 20))
# # colors = ['black', 'white', 'red']
# # cmap = ListedColormap(colors)
# print(datapoint["target_flash_max_average_height"]*100)
# print(datapoint["target_flash_max_length"]*100)
# print(datapoint["target_flash_total_area"]*100*100)
# print(datapoint["bin_num"])
# # for i in range(5):
# #     axs[i, 0].imshow(datapoint["input_rgb"][i, :, :, :].permute(1, 2, 0).numpy())
# #     axs[i, 1].imshow(datapoint["target_segmentation"][i, :, :].numpy(), cmap=cmap, vmin=0, vmax=2)
# #     print(np.unique(datapoint["target_segmentation"][i, :, :].numpy()))
    
# print(datapoint["input_rgb"][0, :, :, :].shape)
# # plt.show()
################################################################################################################################


### Multi-View CNN (MVCNN) Architecture ###

# class MVCNN(nn.Module):
#     def __init__(self, num_classes=100):
#         super(MVCNN, self).__init__()
#         self.num_classes = num_classes
#         resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         fc_in_features = resnet50.fc.in_features
#         self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # As in ResNet
#         self.classifier = nn.Linear(fc_in_features, num_classes)
        

#     def forward(self, inputs):
#         num_views = inputs.size(1)
#         batch_size = inputs.size(0)
#         x = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))
#         x = self.feature_extractor(x)
#         x = x.view(batch_size, num_views, x.shape[-3], x.shape[-2], x.shape[-1])
#         x = torch.max(x, 1)[0]
#         x = self.adaptive_pool(x)
#         x = self.classifier(x.squeeze())
#         return x

# LATE FUSION
class MVCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(MVCNN, self).__init__()
        self.num_classes = num_classes
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        fc_in_features = resnet50.fc.in_features
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc1 = nn.Linear(fc_in_features*5, 1024)  # 5 views, NEED TO MAKE MODULAR
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(1024, num_classes)
        

    def forward(self, inputs):
        num_views = inputs.size(1)
        batch_size = inputs.size(0)
        x = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))
        x = self.feature_extractor(x)
        x = x.view(batch_size, num_views, x.shape[-3], x.shape[-2], x.shape[-1])
        x  = x.view(batch_size, num_views*x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.fc1(x.squeeze())
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



model = MVCNN(num_classes=100)
print(model)

# HYPERPARAMETERS
num_epochs = 100
batch_size = 8
lr = 1e-3
weight_decay = 0.0001
num_classes = 100
loss_fn = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
warmup_epochs = 5
lr_warmup_factor = 1e-3
def lambda1(current_epoch):
    if current_epoch < warmup_epochs:
        return lr_warmup_factor + (1.0 - lr_warmup_factor) * (current_epoch / warmup_epochs)
    return 1.0
scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, min_lr=lr_warmup_factor*lr)

# Dataloaders
available_cores = os.cpu_count()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=available_cores)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  drop_last=True, num_workers=available_cores)

# train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)

### wandb logging ############################################################################################################
wandb_log = True
if wandb_log:
    import wandb
    wandb.init(
    # set the wandb project where this run will be logged
    project="multi-view-classification-basic",

    # track hyperparameters and run metadata
    config={
        "optimizer": "adam",
        "num_classes": num_classes,
        "batch_size": batch_size,
        "learning_rate": lr,
        "loss": "Weighted Cross Entropy",
        # "dataset": "train",
        "epochs": num_epochs,
        "architecture": str(model),
        "img_side": img_size ,
        "schedular": "plateu and warmup",
        }
    )
###################################################################################################################################



### Training Loop ###
torch.cuda.empty_cache()
best_val_accuracy = 0.0
model.to(device)
scaler = GradScaler()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    # train one epoch
    model.train()
    train_loss = 0.0
    train_corrects = 0

    for batch in train_loader:

        input_rgbs = batch["input_rgb"].to(device)
        bin_nums = batch["bin_num"].to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(input_rgbs)
            loss = loss_fn(outputs, bin_nums)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        corrects = (torch.argmax(outputs, 1) == bin_nums).sum().item()
        train_corrects += corrects
        if wandb_log:
            wandb.log({"batch_train_loss": loss.item(), "batch_train_accuracy": corrects/batch_size})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})


    train_loss /= len(train_loader)
    train_accuracy = train_corrects / len(train_loader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    if wandb_log:
        wandb.log({"epoch_train_loss": train_loss, "epoch_train_accuracy": train_accuracy})
    
    ### eval the model
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_corrects = 0
        with autocast():
            for batch in val_loader:
                input_rgbs = batch["input_rgb"].to(device)
                bin_nums = batch["bin_num"].to(device)
                outputs = model(input_rgbs)
                loss = loss_fn(outputs, bin_nums)
                val_loss += loss.item()
                val_corrects += (torch.argmax(outputs, 1) == bin_nums).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = val_corrects / len(val_loader.dataset)
            best_val_accuracy = max(best_val_accuracy, val_accuracy)
            if val_accuracy == best_val_accuracy:
                torch.save(model.state_dict(), "best_model_weighted.pth")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            if wandb_log:
                wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
    if epoch < warmup_epochs:
        scheduler_warmup.step()
    # if epoch >= warmup_epochs:
    #     scheduler_plateau.step(val_accuracy)    