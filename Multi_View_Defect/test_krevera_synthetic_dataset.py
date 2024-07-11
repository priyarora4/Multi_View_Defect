# pylint: skip-file
import os
import matplotlib.pyplot as plt
import pytest
import torch
from krevera_ai.ai.train.datasets.krevera_synthetic_dataset import (
    KreveraSyntheticDataset,
)


@pytest.mark.parametrize(
    "dataset_path",
    [
        ("~/krevera_data/datasets/krevera_synthetic_dataset_20240212_094259"),
    ],
)
def test_shape(dataset_path: str):
    dataset = KreveraSyntheticDataset(dataset_path)

    datapoint = dataset[0]

    # Check keys
    keys_to_check = [
        "input_rgb",
        "target_segmentation",
        "target_flash_max_average_height",
        "target_flash_max_length",
        "target_flash_total_area",
    ]
    for key in keys_to_check:
        assert key in datapoint, f"key: {key} not found in datapoint"

    # Check input_rgb shape
    num_images, num_channels, height, width = datapoint["input_rgb"].shape
    assert num_images == dataset.num_cameras
    assert num_channels == 3
    assert height == 1080
    assert width == 1920

    # Check target_segmentation shape
    num_images, height, width = datapoint["target_segmentation"].shape
    assert num_images == dataset.num_cameras
    assert height == 1080
    assert width == 1920


@pytest.mark.parametrize(
    "dataset_path",
    [
        ("~/krevera_data/datasets/krevera_synthetic_dataset_20240212_094259"),
    ],
)
def test_data_loader(dataset_path: str, batch_size=2):
    data_loader = torch.utils.data.DataLoader(
        KreveraSyntheticDataset(dataset_path),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        multiprocessing_context="spawn",
    )
    for batch in data_loader:
        for key, value in batch.items():
            assert (
                len(value) == batch_size
            ), f"key: {key} has wrong length, {len(value)} expected {batch_size}"
        break


if __name__ == "__main__":
    dataset = KreveraSyntheticDataset(
        "~/krevera_data/datasets/krevera_synthetic_dataset_20240212_094259",
        zero_one_normalize=True,
    )

    print(f"dataset len: {len(dataset)}")

    datapoint = dataset[2]

    num_images, num_channels, height, width = datapoint["input_rgb"].shape
    print(datapoint["input_rgb"].shape)

    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))

    for i in range(num_images):
        axs[0, i].imshow(datapoint["input_rgb"][i, :, :, :].permute(1, 2, 0).numpy())
        axs[1, i].imshow(datapoint["target_segmentation"][i, :, :].numpy())

    plt.show()
