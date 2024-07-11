import json
import os
from typing import Optional

import fsspec
import numpy as np
import torch
from packaging import version
from PIL import Image
from torchvision import datapoints
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F


# TODO: Should support multiple dataset directories
# TODO: For debugging, getitem should also return the path to the datapoint
class KreveraSyntheticDataset(torch.utils.data.Dataset):
    """Krevera pytorch synthetic dataset, constructed by omniverse krevera writer,
    with multiple render products and scene products. Per datapoint this dataset contains
    rgb, instance segmentation, bounding box, and camera data per render product, and
    defect data per scene product.

    Args:
        dataset_directory (str): The directory containing the dataset.
        file_system_protocol (Optional[str], optional): The filesystem to use, when set
            to None, the filesystem is implicitly discovered. Defaults to None.
        zero_one_normalize (bool, optional): Whether to normalize the rgb images to [0, 1].
    """

    def __init__(
        self,
        dataset_directory: str,
        file_system_protocol: Optional[str] = None,
        zero_one_normalize: bool = False,
        is_train: bool = True,
        num_bins = 100, 
        max_flash_area = 50, 
        img_size = 800,
        class_weights_file_path = None,
    ) -> None:
        
        ### Create bins evenly spaced######
        self.num_bins = num_bins
        self.max_flash_area = max_flash_area
        bin_edges = np.linspace(0, self.max_flash_area, self.num_bins)
        self.bin_edges = np.append(bin_edges, np.inf)
        if class_weights_file_path is not None:
            self.class_weights = np.load(class_weights_file_path)
        ###################################
        self.img_size = img_size
        self.is_train = is_train
        self.dataset_directory = os.path.expanduser(dataset_directory)
        self.file_system_protocol = file_system_protocol
        self.zero_one_normalize = zero_one_normalize
        if self.file_system_protocol is None:
            self.file_system_protocol = fsspec.utils.get_protocol(
                self.dataset_directory
            )
        self.file_system = fsspec.filesystem(self.file_system_protocol)
        # TODO: We should read the metadata from the dataset directory and change the dataset class based on the type and version of the dataset

        # Read the metadata from the dataset directory
        metadata = json.load(
            self.file_system.open(os.path.join(self.dataset_directory, "metadata.txt"))
        )
        self.metadata_name = metadata["name"]
        self.metadata_version = version.parse(metadata["version"])
        # Check that this is the correct dataset type
        expected_metadata_name = "KreveraWriter"
        expected_metadata_version = version.parse("0.0.1")
        assert (
            self.metadata_name == expected_metadata_name
        ), f"Invalid dataset type, got {self.metadata_name}, expected {expected_metadata_name}"
        assert (
            self.metadata_version == expected_metadata_version
        ), f"Invalid dataset version, got {self.metadata_version}, expected {expected_metadata_version}"

        self.datapoint_paths = self.file_system.ls(
            os.path.join(self.dataset_directory, "datapoints")
        )
        self.dataset_len = len(self.datapoint_paths)

        # Open the first datapoint to get the render product names
        self.render_product_names = []
        for file_path in self.file_system.ls(self.datapoint_paths[0]):
            file_name = os.path.basename(file_path)
            if file_name.startswith("RenderProduct"):
                self.render_product_names.append(file_name)
        self.num_cameras = len(self.render_product_names)
        self.scene_product_name = "SceneProduct"

        # TODO: make this configurable
        # Create transforms
        self.set_transforms()
        
    def set_transforms(self):
        if not self.is_train:
            self.geo_transform = transforms.Compose([
                transforms.Pad((0, 4, 0, 0), fill=0, padding_mode='constant'),  
                # Resize down by a factor of 2 to 544x960
                transforms.Resize((544, 960)),
                 ])
            
            self.non_geo_transform = transforms.Compose([
                transforms.Normalize(mean=[0.4016, 0.3994, 0.4520], std=[0.2425, 0.2206, 0.1979]),
            ])
        else:
            self.geo_transform = transforms.Compose([
                    transforms.Pad((0, 4, 0, 0), fill=0, padding_mode='constant'), 
                    # Resize down by a factor of 2 to 544x960
                    transforms.Resize((544, 960)),
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                    ),
                    # transforms.ColorJitter(
                    #     brightness=0.25,
                    #     contrast=0.25,
                    #     saturation=0.25,
                    #     hue=0.25,
                    # ),
                    # transforms.RandomApply(
                    #     [transforms.GaussianBlur(kernel_size=11, sigma=(0.5, 2.0))]
                    # ),
                    transforms.RandomApply(
                        [transforms.ElasticTransform(alpha=[1.0, 30.0])]
                    ),
                    # transforms.ToImageTensor(),
                    # transforms.Normalize(mean=[0.4016, 0.3994, 0.4520], std=[0.2425, 0.2206, 0.1979]),
                ]
            )
            self.non_geo_transform = transforms.Compose([
                transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.25,
                        hue=0.25,
                    ),
                transforms.Normalize(mean=[0.4016, 0.3994, 0.4520], std=[0.2425, 0.2206, 0.1979]),
            ])

    def __len__(self):
        return self.dataset_len

    def _read_raw_data(
        self, datapoint_path: str
    ) -> tuple[list, list, list, list, list, list, list, dict]:
        # Per render product read the raw data from storage
        rgb_image_list = []
        instance_segmentation_image_list = []
        instance_segmentation_semantics_mapping_list = []
        bounding_box_2d_tight_list = []
        bounding_box_2d_tight_labels_list = []
        bounding_box_2d_tight_prim_paths_list = []
        camera_params_list = []
        for render_product_name in self.render_product_names:
            render_product_path = os.path.join(datapoint_path, render_product_name)

            # rgb:
            rgb_image_list.append(
                torch.from_numpy(
                    np.array(  # NOTE np.asarray would be better (no copy) but the array is unwriteable
                        Image.open(
                            self.file_system.open(
                                os.path.join(render_product_path, "rgb.png"), "rb"
                            )
                        )
                    )
                )
            )
            # instance segmentation:
            instance_segmentation_image_list.append(
                torch.from_numpy(
                    np.array(  # NOTE np.asarray would be better (no copy) but the array is unwriteable
                        Image.open(
                            self.file_system.open(
                                os.path.join(
                                    render_product_path, "instance_segmentation.png"
                                ),
                                "rb",
                            )
                        )
                    )
                )
            )
            instance_segmentation_semantics_mapping_list.append(
                json.load(
                    self.file_system.open(
                        os.path.join(
                            render_product_path,
                            "instance_segmentation_semantics_mapping.json",
                        )
                    )
                )
            )
            # bounding box:
            bounding_box_2d_tight_list.append(
                np.load(
                    self.file_system.open(
                        os.path.join(render_product_path, "bounding_box_2d_tight.npy"),
                        "rb",
                    )
                )
            )
            bounding_box_2d_tight_labels_list.append(
                json.load(
                    self.file_system.open(
                        os.path.join(
                            render_product_path, "bounding_box_2d_tight_labels.json"
                        )
                    )
                )
            )
            bounding_box_2d_tight_prim_paths_list.append(
                json.load(
                    self.file_system.open(
                        os.path.join(
                            render_product_path,
                            "bounding_box_2d_tight_prim_paths.json",
                        )
                    )
                )
            )
            # camera params:
            camera_params_list.append(
                json.load(
                    self.file_system.open(
                        os.path.join(render_product_path, "camera_params.json")
                    )
                )
            )
        # Read the raw scene product data from storage
        flash_defects = json.load(
            self.file_system.open(
                os.path.join(
                    datapoint_path, self.scene_product_name, "flash_defects.json"
                )
            )
        )
        return (
            rgb_image_list,
            instance_segmentation_image_list,
            instance_segmentation_semantics_mapping_list,
            bounding_box_2d_tight_list,
            bounding_box_2d_tight_labels_list,
            bounding_box_2d_tight_prim_paths_list,
            camera_params_list,
            flash_defects,
        )

    def _compute_input_rgb(
        self,
        rgb_image_list: list[torch.Tensor],
    ) -> torch.Tensor:
        # Stack the rgb images remove the alpha channel and add a batch dimension
        return torch.stack(rgb_image_list).permute(0, 3, 1, 2)[:, :3, :, :]

    def _compute_gt_segmentation(
        self,
        instance_segmentation_image_list: list[torch.Tensor],
        instance_segmentation_semantics_mapping_list: list[dict],
    ) -> torch.Tensor:
        # Extract the gt segmentation from the instance segmentation images
        gt_segmentation = torch.zeros(
            (
                len(instance_segmentation_image_list),
                instance_segmentation_image_list[0].shape[0],
                instance_segmentation_image_list[0].shape[1],
            ),
            dtype=torch.uint8,
        )
        for i, (
            instance_segmentation_image,
            instance_segmentation_semantics_mapping,
        ) in enumerate(
            zip(
                instance_segmentation_image_list,
                instance_segmentation_semantics_mapping_list,
            )
        ):
            inspection_object_key = None
            flash_keys = []
            # Find class inspection_object
            for (
                seg_key,
                seg_value,
            ) in instance_segmentation_semantics_mapping.items():
                class_name = seg_value.get("class", None)
                if class_name == "inspection_object":
                    inspection_object_key = int(seg_key)
                if class_name == "flash,inspection_object":
                    flash_keys.append(int(seg_key))
            assert (
                inspection_object_key is not None
            ), f"Could not find inspection_object in instance segmentation semantics mapping: {instance_segmentation_semantics_mapping}"
            gt_segmentation[i, instance_segmentation_image == inspection_object_key] = 1
            if len(flash_keys) > 0:
                flash_mask = torch.full_like(
                    instance_segmentation_image, False, dtype=torch.bool
                )
                for flash_key in flash_keys:
                    # TODO: Check size and or occlusion of flash, and ignore if too small or occluded
                    flash_mask = flash_mask | (instance_segmentation_image == flash_key)
                gt_segmentation[i, flash_mask] = 2
        return gt_segmentation

    def _compute_global_flash_defects(
        self,
        flash_defects: dict,
        bounding_box2d_tight_list: list[np.ndarray],
        bounding_box_2d_tight_prim_paths_list: list[list],
        occlusion_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: occlusion goes from [0, 1] or -1 for unknown
        flash_defects_occlusion = {key: 1 for key in flash_defects.keys()}
        for (
            bounding_box_2d_tight,
            bounding_box_2d_tight_prim_paths,
        ) in zip(
            bounding_box2d_tight_list,
            bounding_box_2d_tight_prim_paths_list,
        ):
            for data, prim_path in zip(
                bounding_box_2d_tight,
                bounding_box_2d_tight_prim_paths,
            ):
                if prim_path in flash_defects:
                    occlusion = data[
                        -1
                    ]  # NOTE: [('semanticId', '<u4'), ('x_min', '<i4'), ('y_min', '<i4'), ('x_max', '<i4'), ('y_max', '<i4'), ('occlusionRatio', '<f4')]
                    # Treat unknown occlusion as fully occluded
                    if occlusion == -1:
                        occlusion = 1
                    flash_defects_occlusion[prim_path] = min(
                        flash_defects_occlusion[prim_path], occlusion
                    )
        max_average_height = 0
        max_length = 0
        total_area = 0
        for prim_path, occlusion in flash_defects_occlusion.items():
            if occlusion > occlusion_threshold:
                continue
            max_average_height = max(
                max_average_height,
                flash_defects[prim_path]["defect_flash_average_height"],
            )
            max_length = max(
                max_length, flash_defects[prim_path]["defect_flash_length"]
            )
            total_area += flash_defects[prim_path]["defect_flash_area"]

        return (
            torch.tensor(max_average_height, dtype=torch.float32),
            torch.tensor(max_length, dtype=torch.float32),
            torch.tensor(total_area, dtype=torch.float32),
        )

    def __getitem__(self, idx):
        # try:
            (
                rgb_image_list,
                instance_segmentation_image_list,
                instance_segmentation_semantics_mapping_list,
                bounding_box_2d_tight_list,
                _,  # bounding_box_2d_tight_labels_list,
                bounding_box_2d_tight_prim_paths_list,
                _,  # camera_params_list,
                flash_defects,
            ) = self._read_raw_data(self.datapoint_paths[idx])

            # Extract global flash defects data
            (
                flash_defects_max_average_height,
                flash_defects_max_length,
                flash_defects_total_area,
            ) = self._compute_global_flash_defects(
                flash_defects,
                bounding_box_2d_tight_list,
                bounding_box_2d_tight_prim_paths_list,
            )

            # Extract the gt segmentation from the instance segmentation images
            gt_segmentation = self._compute_gt_segmentation(
                instance_segmentation_image_list,
                instance_segmentation_semantics_mapping_list,
            )

            input_rgb = self._compute_input_rgb(rgb_image_list)
            input_rgb = input_rgb.float() / 255.0
            input_rgb, gt_segmentation = self.geo_transform(
                input_rgb, datapoints.Mask(gt_segmentation)
            )
            input_rgb = self.non_geo_transform(input_rgb)
            
            flash_defects_total_area_cm = flash_defects_total_area*100*100
            bin_num = np.digitize(flash_defects_total_area_cm, self.bin_edges, right=False) - 1
            bin_num = torch.tensor(bin_num)
            # if self.zero_one_normalize:
            #     input_rgb = input_rgb / 255.0
            gt_segmentation = gt_segmentation.long()
            gt_segmentation = F.one_hot(gt_segmentation, num_classes=3).permute(0, 3, 1, 2).float()
            return {
                "input_rgb": input_rgb,
                "target_segmentation": gt_segmentation,
                "target_flash_max_average_height": flash_defects_max_average_height,
                "target_flash_max_length": flash_defects_max_length,
                "target_flash_total_area": flash_defects_total_area,
                "bin_num": bin_num
            }
        # except Exception as e:
        #     print(f"Error loading datapoint {idx}: {self.datapoint_paths[idx]}, {e}")
        #     new_idx = np.random.randint(low=0, high=self.dataset_len)
        #     print(f"Loading random datapoint: {new_idx}")
        #     return self.__getitem__(new_idx)


# # TODO: Should support multiple dataset directories
# # TODO: For debugging, getitem should also return the path to the datapoint
# class KreveraSyntheticDatasetLegacy(torch.utils.data.Dataset):
#     """Krevera pytorch synthetic dataset, constructed by omniverse basic writer, with multiple render products.
#     This dataset contains muliple rgb images and instance segmentation images per datapoint.

#     Args:
#         dataset_directory (str): The directory containing the dataset.
#         file_system_protocol (Optional[str], optional): The filesystem to use, when set
#             to None, the filesystem is implicitly discovered. Defaults to None.
#         zero_one_normalize (bool, optional): Whether to normalize the rgb images to [0, 1].
#     """

#     def __init__(
#         self,
#         dataset_directory: str,
#         file_system_protocol: Optional[str] = None,
#         zero_one_normalize: bool = False,
#     ) -> None:
#         self.dataset_directory = os.path.expanduser(dataset_directory)
#         self.file_system_protocol = file_system_protocol
#         self.zero_one_normalize = zero_one_normalize
#         if self.file_system_protocol is None:
#             self.file_system_protocol = fsspec.utils.get_protocol(
#                 self.dataset_directory
#             )
#         self.file_system = fsspec.filesystem(self.file_system_protocol)

#         # Get the list of camera directories that follow the pattern RenderProduct*
#         self.camera_dirs = [
#             d
#             for d in self.file_system.ls(self.dataset_directory)
#             if os.path.basename(d).startswith("RenderProduct")
#         ]
#         assert len(self.camera_dirs) > 0, "No camera directories found"

#         self.rgb_dirs = [
#             os.path.join(camera_dir, "rgb") for camera_dir in self.camera_dirs
#         ]
#         self.instance_segmentation_dirs = [
#             os.path.join(camera_dir, "instance_segmentation")
#             for camera_dir in self.camera_dirs
#         ]

#         # Get the datset len from each camera directory by looking in the rgb directory
#         self.dataset_lens = [
#             len(self.file_system.ls(os.path.join(camera_dir, "rgb")))
#             for camera_dir in self.camera_dirs
#         ]
#         # Assert that all the datasets are the same length
#         assert all(
#             [length == self.dataset_lens[0] for length in self.dataset_lens]
#         ), f"Dataset lengths are not the same for all cameras: {self.dataset_lens}"

#         self.dataset_len = self.dataset_lens[0]

#         # Create a list of tuples of the rgb and instance segmentation file paths per camera per dataset index
#         self.dataset_paths = []
#         for i in range(self.dataset_len):
#             # rgb file pattern: rgb_0000.png
#             # instance segmentation file pattern: instance_segmentation_0000.png
#             # instance segmentation semantics mapping file pattern: instance_segmentation_semantics_mapping_0000.json
#             datapoint_paths = {
#                 "rgb": [],
#                 "instance_segmentation": [],
#                 "instance_segmentation_semantics_mapping": [],
#             }
#             for rgb_dir, instance_segmentation_dir in zip(
#                 self.rgb_dirs, self.instance_segmentation_dirs
#             ):
#                 datapoint_paths["rgb"].append(os.path.join(rgb_dir, f"rgb_{i:04d}.png"))
#                 datapoint_paths["instance_segmentation"].append(
#                     os.path.join(
#                         instance_segmentation_dir, f"instance_segmentation_{i:04d}.png"
#                     )
#                 )
#                 datapoint_paths["instance_segmentation_semantics_mapping"].append(
#                     os.path.join(
#                         instance_segmentation_dir,
#                         f"instance_segmentation_semantics_mapping_{i:04d}.json",
#                     )
#                 )
#             self.dataset_paths.append(datapoint_paths)

#         # TODO: make this configurable
#         # Create transforms
            
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomAffine(
#                     degrees=10,
#                     translate=(0.1, 0.1),
#                     scale=(0.9, 1.1),
#                 ),
#                 transforms.ColorJitter(
#                     brightness=0.5,
#                     contrast=0.5,
#                     saturation=0.5,
#                     hue=0.3,
#                 ),
#                 transforms.RandomApply(
#                     [transforms.GaussianBlur(kernel_size=11, sigma=(0.5, 2.0))]
#                 ),
#                 transforms.RandomApply(
#                     [transforms.ElasticTransform(alpha=[1.0, 30.0])]
#                 ),
#             ]
#         )

#     def __len__(self):
#         return self.dataset_len

#     def __getitem__(self, idx):
#         try:
#             datapoint_paths = self.dataset_paths[idx]

#             # Read raw data from disk:
#             rgb_images = [
#                 torch.from_numpy(
#                     np.asarray(
#                         Image.open(self.file_system.open(rgb_image_path, "rb"))
#                     ).copy()
#                 )
#                 for rgb_image_path in datapoint_paths["rgb"]
#             ]
#             instance_segmentation_images = [
#                 torch.from_numpy(
#                     np.asarray(
#                         Image.open(
#                             self.file_system.open(
#                                 instance_segmentation_image_path, "rb"
#                             )
#                         )
#                     ).copy()
#                 )
#                 for instance_segmentation_image_path in datapoint_paths[
#                     "instance_segmentation"
#                 ]
#             ]
#             instance_segmentation_semantics_mappings = [
#                 json.load(
#                     self.file_system.open(instance_segmentation_semantics_mapping_path)
#                 )
#                 for instance_segmentation_semantics_mapping_path in datapoint_paths[
#                     "instance_segmentation_semantics_mapping"
#                 ]
#             ]

#             # Extract the gt segmentation from the instance segmentation images
#             gt_segmentation = torch.zeros(
#                 (
#                     len(instance_segmentation_images),
#                     instance_segmentation_images[0].shape[0],
#                     instance_segmentation_images[0].shape[1],
#                 ),
#                 dtype=torch.uint8,
#             )
#             for i, instance_segmentation_semantics_mapping in enumerate(
#                 instance_segmentation_semantics_mappings
#             ):
#                 inspection_object_key = None
#                 flash_keys = []
#                 # Find class inspection_object
#                 for (
#                     seg_key,
#                     seg_value,
#                 ) in instance_segmentation_semantics_mapping.items():
#                     class_name = seg_value.get("class", None)
#                     if class_name == "inspection_object":
#                         inspection_object_key = int(seg_key)
#                     if class_name == "flash,inspection_object":
#                         flash_keys.append(int(seg_key))
#                 assert (
#                     inspection_object_key is not None
#                 ), f"Could not find inspection_object in instance segmentation semantics mapping: {instance_segmentation_semantics_mapping}"
#                 gt_segmentation[
#                     i, instance_segmentation_images[i] == inspection_object_key
#                 ] = 1
#                 if len(flash_keys) > 0:
#                     flash_mask = torch.full_like(
#                         instance_segmentation_images[i], False, dtype=torch.bool
#                     )
#                     for flash_key in flash_keys:
#                         flash_mask = flash_mask | (
#                             instance_segmentation_images[i] == flash_key
#                         )
#                     gt_segmentation[i, flash_mask] = 2
#                     # gt_segmentation[i, instance_segmentation_images[i] == flash_keys] = 2

#             # Stack the rgb images remove the alpha channel and add a batch dimension
#             # rgb_images = torch.stack(rgb_images).permute(0, 3, 1, 2)[:, :3, :, :].float() / 255.0
#             rgb_images = torch.stack(rgb_images).permute(0, 3, 1, 2)[:, :3, :, :]
#             rgb_images, gt_segmentation = self.transform(
#                 rgb_images, datapoints.Mask(gt_segmentation)
#             )
#             rgb_images = rgb_images.float()
#             if self.zero_one_normalize:
#                 rgb_images = rgb_images / 255.0
#             return rgb_images, gt_segmentation.long()
#         except Exception as e:
#             print(f"Error loading datapoint: {idx}, {e}")
#             new_idx = np.random.randint(low=0, high=self.dataset_len)
#             print(f"Loading random datapoint: {new_idx}")
#             return self.__getitem__(new_idx)
