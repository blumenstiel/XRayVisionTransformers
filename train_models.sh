#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH --output=train_models.out


# Activate Anaconda work environment
source activate Xray

python train.py -config configs/vit_small.yaml MODEL.DIR results/vit_small DEVICE cuda:0 &
python train.py -config configs/vit_base_patch32.yaml MODEL.DIR results/vit_base_patch32 DEVICE cuda:1 &
wait
python train.py -config configs/vit_base.yaml MODEL.DIR results/vit_base DEVICE cuda:0 &
python train.py -config configs/vit_base_224.yaml MODEL.DIR results/vit_base_224 DEVICE cuda:1 &
wait
python train.py -config configs/deit_base.yaml MODEL.DIR results/deit_base DEVICE cuda:0 &
python train.py -config configs/swin_base.yaml MODEL.DIR results/swin_base DEVICE cuda:1 &
wait
python train.py -config configs/vit_base_r50.yaml MODEL.DIR results/vit_base_r50 DEVICE cuda:0 &
python train.py -config configs/resnet_50x1.yaml MODEL.DIR results/resnet_50x1 DEVICE cuda:1 &
wait
python train.py -config configs/vit_large.yaml MODEL.DIR results/vit_large DEVICE cuda:0 &
python train.py -config configs/resnet_152x2.yaml MODEL.DIR results/resnet_152x2 DEVICE cuda:1 &
wait

# sbatch -p sdil -t 24:00:00 train_models.sh