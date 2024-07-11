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
import torch.nn.functional as F

from krevera_project_segment import NestedUNet


def inference_and_plot(datapoint, model, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        views = datapoint['input_rgb']
        target_segmentation = datapoint['target_segmentation']
        views = views.to(device)
        outputs = model(views)
        
        predicted_segmentation = torch.argmax(outputs, dim=1)
         # Visualize data
        mean = np.array([0.4016, 0.3994, 0.4520])
        std = np.array([0.2425, 0.2206, 0.1979])
        fig, axs = plt.subplots(5, 3, figsize=(15, 25))  # Prepare subplots for 5 data points
        colors = ['black', 'white', 'red']
        cmap = ListedColormap(colors)

        for i in range(5):
            # Normalize and display the input images
            img = views[i].cpu().permute(1, 2, 0).numpy() * std + mean
            axs[i, 0].imshow(img)
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Input Image')

            # Display the ground truth segmentation
            ground_truth = torch.argmax(datapoint['target_segmentation'], dim=1)[i].cpu().numpy()
            axs[i, 1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=2)
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Ground Truth')

            # Display the predicted segmentation
            prediction = predicted_segmentation[i].cpu().numpy()
            axs[i, 2].imshow(prediction, cmap=cmap, vmin=0, vmax=2)
            axs[i, 2].axis('off')
            axs[i, 2].set_title('Predicted Segmentation')

        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = NestedUNet(num_classes=3, input_channels=3, deep_supervision=False)
    model.load_state_dict(torch.load('NestedUnet_Segmentation_best.pth'))
    print(model)

    test_dataset = KreveraSyntheticDataset("./krevera_synthetic_dataset_test", 
                                    zero_one_normalize=True, 
                                    is_train=False
                                    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  drop_last=True, num_workers=2)

    datapoint = test_dataset[17]
    inference_and_plot(datapoint, model, device)