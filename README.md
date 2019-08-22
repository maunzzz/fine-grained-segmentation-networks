# Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization
This is an implementation of the work published in Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization (https://arxiv.org/abs/1908.06387)

## Resources 
The datasets used in the paper is published at visuallocalization.net

## Trained Models
https://drive.google.com/open?id=14joxT0XFreW1WX3M8oTiCV69hZTiJTMV

## Installation
A Dockerfile is provided, either build a docker image using this or refer to the requirements listed in the file.
In addition, a requirements.txt is provided.

## Usage 
- Download Cityscapes and Mapillary Vistas
- Use /utils/convert_vistas_to_cityscapes.py to create cityscapes class annotations for the Vistas images
- Download the correspondence datasets
- Download the images associated with the correspondence datasets (instructions available in dataset readme)
- Create a global_otps.json and set the paths (see global_opts_example.json)
- Get base models from the trained models link above, place the 'base-networks' folder in global_opts['result-folder']
- Run /clustering/setup_cluster_dataset for the dataset to be trained on
- Train, see train/train_many_cluster.py for reproduction of paper main experiments

## Reference
If you use this code, please cite the following paper:

Måns Larsson, Erik Stenborg, Carl Toft, Lars Hammarstrand, Torsten Sattler and Fredrik Kahl
"Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization" Proc. ICCV (2019).

```
@InProceedings{larsson2019fgsn,
  author = {Larsson, M{\aa}ns and Stenborg, Erik and Toft, Carl and Hammarstrand, Lars and Sattler, Torsten and Kahl, Fredrik},
  title = {Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year = {2019}
} 
```

## Other
Some code from https://github.com/facebookresearch/deepcluster, https://github.com/zijundeng/pytorch-semantic-segmentation, and https://github.com/kazuto1011/pspnet-pytorch was used.
