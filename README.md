# XRayVisionTransformers
PyTorch implementation for the student project: [Exploring Possibilities of Vision Transformers for X-Ray Images]().

## Installation
Run the following command to create a virtual environment and install the required packages:
```
conda create -f environment.yml
```
Activate the environment with:
```
conda activate Xray
```

## Dataset
The dataset is available from https://stanfordmlgroup.github.io/competitions/mura but requires registration.
Place the unzipped folder "MURA-v1.1" in the root directory.

## Usage
To train a model, run the following command:
```
python train.py --config configs/vit_base.yaml
```

To evaluate a model, run the following command:
```
python evaluate.py --config configs/vit_base.yaml
```

Pretrained models are downloaded from [timm PyTorch image models](https://github.com/rwightman/pytorch-image-models).
See [Imagenet results](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv) for available models and consider information about the [pre-training dataset](https://github.com/rwightman/pytorch-image-models/blob/master/results/model_metadata-in1k.csv).

The model name, output dir, input size, and standardization values have to be specified, see [src/config.py](https://github.com/blumenstiel/XRayVisionTransformers/blob/main/src/config.py) for details.

Tensorboard logs can be accessed with the following command:
```
tensorboard --logdir tensorboard
```

## Citation


### BibTex

```
@misc{blumenstiel2022xray,
  author = {Benedikt Blumenstiel},
  title = {Exploring Possibilities of Vision Transformers for X-Ray Images},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/blumenstiel/XRayVisionTransformers}
}
```
