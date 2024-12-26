# AttenScribble

This repository is the official implementation of the paper AttenScribble: Attentive Similarity Learning for Scribble-Supervised Medical Image Segmentation

## Datasets

### ACDC
1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
2. The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 
3. The pre-processed ACDC data used for training could be directly downloaded from [ACDC_dataset](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).

The processed ACDC data also can be downloaded from [Google drive](https://drive.google.com/drive/folders/1bGmNCbNPrMuFqWVSQeR_1tDsrB7nKFAu?usp=drive_link).
## Requirements

Some important required packages include:
* Python 3.7
* CUDA 11.8
* [Pytorch](https://pytorch.org) 1.9.0.
* torchvision 0.10.0
* Some basic python packages such as Numpy, Scikit-image, SimpleITK......

## Usage
1. Clone this project.
   ```
   git clone https://github.com/YangQinzhu/AttenScribble.git
   cd AttenScribble
   ```

2. Data pre-processing os used or the processed data.
   
   Download data [Google drive](https://drive.google.com/drive/folders/1bGmNCbNPrMuFqWVSQeR_1tDsrB7nKFAu?usp=drive_link) and put it into the folder `data/ACDC/`.

3. Train the model
    ```
    sh run_acdc.sh
    ```
    
    Please change the file path as you set.

4. Test the model
    ```
    sh test_acdc.sh
    ```

# Question
Please open an issue or email qinzhuyang@foxmail.com / kevinmtian@gmail.com for any questions.


# Acknowledgement
The codebase is adapted from [WSL4MIS](https://github.com/HiLab-git/WSL4MIS).
