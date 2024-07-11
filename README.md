# Multi_View_Defect

In Multi_View_Defect folder I have provided some of the code files that should be able to run very easily. 

`krevera_synthetic_dataset.py` is a slightly modified dataset class.  

`krevera_project_segment.py` is the training script for the Nested UNet model. The code is pretty self contained so there's no need to jump to many other files. You can simply read from top to bottom as this script is formatted in a similar way to a python notebook. I tried to make it simple to understand. 

`krevera_project_multi_view_unet.py` is the training script file for the multi-view CNN with NestedUNet as the backbone feature extractor. It is in a similar format to the segment script so you can simply read from top to bottom. 

`krevera_project_segment_test.ipynb` is a python notebook that has inference scripts for the trained Nested UNet model NestedUnet_Segmentation_best.pth
