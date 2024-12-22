# AttenScribble

This repository is the official implementation of the paper AttenScribble: Attentive Similarity Learning for Scribble-Supervised Medical Image Segmentation

## Datasets

### ACDC
1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
2. The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 
3. The pre-processed ACDC data used for training could be directly downloaded from [ACDC_dataset](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).

## Requirements

Some important required packages include:
* Python 3.7
* CUDA 11.8
* [Pytorch](https://pytorch.org) 1.9.0.
* torchvision 0.10.0
* Some basic python packages such as Numpy, Scikit-image, SimpleITK......

## Training

To train the model, run this command:

```
sh run_acdc.sh
```
Please change the file path as you set.

## Evaluation

To evaluate the model, run this command:

```eval
test_acdc.sh
```

# Todo...
