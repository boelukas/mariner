# MaRINeR: Enhancing Novel Views by Matching Rendered Images with Nearby References
[Project Page](https://boelukas.github.io/mariner/) | [Paper](http://arxiv.org/abs/2407.13745)
<!-- TODO: | [Paper]() | [Video]() -->

This repository contains code for the paper "MaRINeR: Enhancing Novel Views by Matching Rendered Images with Nearby References", ECCV 2024.

## Table of Content
- [Demo](#demo)
- [Installation](#installation)
- [Data and model weights](#data-and-model-weights)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)
<!-- TODO: - [Citation](#citation) -->

## Demo
We provide a demo using Google Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boelukas/mariner/blob/main/notebooks/mariner_demo.ipynb)

## Installation


[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) **(Recommended)** - Clone the repository and then create and activate a `mariner` conda environment using the provided environment definition:

```shell
conda env create -f environment.yml
conda activate mariner
```
### Dependencies
-  torchvision==0.16.0
- tensorboard===2.17.0
- lightning==2.2.1
- opencv-python==4.10.0.84
- jsonargparse==4.31.0
- jsonargparse[signatures]>=4.26.1
- erqa==1.1.2
- lpips==0.1.4
- tabulate==0.9.0
- rich==13.7.1
- numpy==1.26.4

## Data and model weights
We train and test our model on data from [LaMAR](https://github.com/microsoft/lamar-benchmark). Additionally we test the model on data from [MeshLoc](https://data.ciirc.cvut.cz/public/projects/2022MeshLoc/) (renderings), [12Scenes](https://graphics.stanford.edu/projects/reloc/#data), [NeRF](https://exp-deeplearning-tools.github.io/nerf/), [IBRnet](https://ibrnet.github.io/) (renderings), [Nerfacto](https://docs.nerf.studio/nerfology/methods/nerfacto.html).

- [Demo dataset](https://drive.google.com/file/d/1VmhgXL1IFRwDlCSPZcwTt9ZsKorSknKk/view?usp=drive_link)
- [Train dataset](https://drive.google.com/file/d/1x9Q6np6VklEthr5f3Ne15pUzfcc7Megk/view?usp=drive_link)
- [Test datasets](https://drive.google.com/file/d/1fkajRAyxsaOsCPxZLDU1iUMo8BYZNGej/view?usp=drive_link)
- Download the [weights](https://drive.google.com/file/d/1zb90JWtX5-Si7MklJMqWn1Kwnqsi6mhF/view?usp=drive_link) and place them in pretrained_weights/

## Usage
### Prediction
```shell
python mariner/main.py predict \
    -c configs/MaRINeR.yml \
    --ckpt_path /path/to/the/project/pretrained_weights/mariner.ckpt \
    --data_dir /path/to/the/demo_data
```

The `data_dir` folder needs to contain `input` and `ref` subfolders with the corresponding images.
Creates the outputs in a subfolder `out` in `data_dir`.
```shell
/data_dir/
├── input
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
└── ref
    ├── 00000.png
    ├── 00001.png
    ├── 00002.png
```

### Optional: Larger resolution
For images with resolution much larger than 160, [Real-ERSGAN](https://github.com/xinntao/Real-ESRGAN) can be used as pipeline extension.

Pull the submodule and download the pretrained weights:
```shell
git submodule update --init --recursive
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P thirdparty/Real-ESRGAN/weights
cd thirdparty/Real-ESRGAN && python setup.py develop
```
Predict with:
```shell
python scripts/predict_scaled.py \
     -c configs/MaRINeR.yml \
     --ckpt_path /path/to/the/project/pretrained_weights/mariner.ckpt \
     --data_dir /path/to/the/demo_data
```
## Training
To train the model with the default config, 32 GB of VRAM are needed. Parameters, such as the `batch_size`, can be changed in configs/MaRINeR.yml or added as options e.g `--batch_size 2`.
```shell
python mariner/main.py fit \
     -c configs/MaRINeR.yml \
     --train_data_dir /path/to/the/train_data/CAB_merged_LIN/train \
     --test_data_dir /path/to/the/train_data/CAB_merged_LIN/test
```

## Evaluation
Download the test datasets and select one of the available datasets for evaluation: 

`dataset = { CAB_ref_gt, CAB_ref_lvl_1, LIN_ref_lvl_1, HGE_ref_lvl_1, IBRnet, 12SCENES_apt_1_living_ref_lvl_10, NeRF}`
```shell
python scripts/eval_metrics.py \
    --images /path/to/the/test_data/dataset/out \
            /path/to/the/test_data/dataset/gt
```

## Citation
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{bösiger2024marinerenhancingnovelviews,
      title={MaRINeR: Enhancing Novel Views by Matching Rendered Images with Nearby References}, 
      author={Lukas Bösiger and Mihai Dusmanu and Marc Pollefeys and Zuria Bauer},
      year={2024},
      booktitle={European Conference on Computer Vision (ECCV)},
      url={https://arxiv.org/abs/2407.13745}, 
}
```

## Acknowledgement
We borrow some code from [MASA-SR](https://github.com/dvlab-research/MASA-SR).