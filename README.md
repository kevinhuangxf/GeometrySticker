# GeometrySticker

GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields (ECCV 2024)

Paper: [arXiv](https://kevinhuangxf.github.io/GeometrySticker/)

Project Page: [https://kevinhuangxf.github.io/GeometrySticker/](https://kevinhuangxf.github.io/GeometrySticker/)

## Clone this repository

```
git clone --branch main --single-branch https://github.com/kevinhuangxf/GeometrySticker.git
```

## Installation

```
# create conda environment

conda create -n geosticker python=3.8

# install pytorch dependencies

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install other dependencies

pip install requirements.txt
```

## Dataset

Please download the Blender and LLFF datasets from this link: [NeRF datasets](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

## Run experiments

Run GeometrySticker with NGP-based NeRF.

```
cd exp/ngp
```

Train a nerf model.

```
ROOT_DIR=path/to/Synthetic_NeRF

python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name Lego \
    --num_epochs 30 --batch_size 16384 --lr 2e-2 --eval_lpips
```

Train GeometrySticker.

```
python train_geometrysticker.py \
    --root_dir $ROOT_DIR/Lego_geosticker/ \
    --exp_name Lego \
    --lr 1e-4 \
    --num_epochs 5 \
    --weight_path ckpts/nsvf/Lego/epoch=29_slim.ckpt \
    --downsample 0.25
```

## Evaluation

```
# Evaluating on recoloring

ROOT_DIR=path/to/Synthetic_NeRF

python train_geometrysticker.py \
    --root_dir $ROOT_DIR/Lego_geosticker \
    --exp_name Lego_geosticker \
    --weight_path ckpts/nsvf/Lego_geosticker/epoch=4_slim.ckpt \
    --downsample 0.25 \
    --val_only
```

## Ciatation

```
@article{huang2024geometrysticker,
  title     = {GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields},
  author    = {Xiufeng Huang, Ka Chun Cheung, Simon See, Renjie Wan},
  journal   = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```
