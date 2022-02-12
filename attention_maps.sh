#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=attention_maps.out

# Activate Anaconda work environment
source activate Xray

python attention_maps.py -config configs/vit_small.yaml MODEL.DIR results/vit_small
python attention_maps.py -config configs/vit_base.yaml MODEL.DIR results/vit_base
python attention_maps.py -config configs/vit_large.yaml MODEL.DIR results/vit_large

# sbatch -p sdil -t 20:00 attention_maps.sh