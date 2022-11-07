# DUPnet-Pytorch
A water body segmentation method based on dense blocks and the multi-scale pyramid pooling module (DUPnet) are proposed. We analyzed the water segmentation performance of dupnet on two datasets: the LR dataset and the 2020 Gaofen challenge water body segmentation dataset.

1. we create a new dataset called Landsat River dataset (LR dataset) to evaluate the per-formance of the proposed network. Images from Landsat 8 satellite were downloaded freely from the USGS website (https://earthexplorer.usgs.gov/, accessed on 24 August 2022). The LR dataset contains 7,154 images with the size of 128×128.This dataset was partitioned into training, validation, and test sets with a scale of 6:2:2. 

16 mislabeled images: 20172.png  20358.png  20359.png  20420.png  20894.png  21012.png  21037.png  22386.png  22661.png  22898.png  23822.png  23972.png  25054.png  25154.png  25585.png  26551.png  

Link:https://drive.google.com/file/d/1W_N17aHCu8fK1XU8zUEYFkv_cG0yE0OC/view?usp=share_linkg

2. The dataset contains 1000 RGB images from the GF-2 satellite with an image size of 492 x 492. We expanded to 8,000 images by rotating, blurring, brightening, darkening, and adding noise. This dataset was partitioned into training, validation, and test sets with a scale of 6:2:2. 

Link:https://drive.google.com/file/d/1oHyzNfHe_F3MeeUQUoni9dh1LFI_N6RS/view?usp=sharing


Citation: Liu, Z.; Chen, X.; Zhou, S.; Yu, H.; Guo, J.; Liu, Y. DUPnet: Water Body Segmentation with Dense Block and Multi-Scale Spatial Pyramid Pooling for Remote Sensing Images. Remote Sens. 2022, 14, 5567.
