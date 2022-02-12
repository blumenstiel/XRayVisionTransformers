from yacs.config import CfgNode as CN

cfg = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

cfg.MODEL = CN()
# model name
cfg.MODEL.NAME = 'vit_base_patch16_384'
# output directory
cfg.MODEL.DIR = 'results/vit_base_patch16_384'
# name of the model file
cfg.MODEL.FILE = 'best_model.pt'
# model params
cfg.MODEL.INPUT_SIZE = 384
cfg.MODEL.MEAN = [0.5, 0.5, 0.5]
cfg.MODEL.STD = [0.5, 0.5, 0.5]
# whether to average the logits, the probabilities or the embeddings of the different views of a study
cfg.MODEL.AVERAGE = 'logits'
# use two linear layers as head for the classification
cfg.MODEL.NONLINEAR_HEAD = False
cfg.MODEL.NONLINEAR_HEAD_SIZE = 768
# use dropout and stochastic depth regularization
cfg.MODEL.REGULARIZATION = 0.

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

cfg.DATASET = CN()
cfg.DATASET.NAME = 'MURA'
# 0: negative/normal, 1: positive/abnormal
cfg.DATASET.NUM_LABELS = 2
# list of selected extremities
cfg.DATASET.EXTREMITIES = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']
# select data augmentation mode from 'none', 'light', 'medium', 'heavy'
cfg.DATASET.RAND_AUGMENT = 'light'
cfg.DATASET.SINGLE_VIEW = False

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

cfg.TRAIN = CN()
# number of epochs to train
cfg.TRAIN.EPOCHS = 30
# Batch size (ViT_base with batch size 16 requires 32 Gb of RAM while training)
cfg.TRAIN.BATCH_SIZE = 16
# optimizer with parameters
cfg.TRAIN.OPTIMIZER = 'Adam'
# learning rate will be scaled by lr * batch_size / 512
cfg.TRAIN.LEARNING_RATE = 3e-4
# learning rate parameters
cfg.TRAIN.WEIGHT_DECAY = 0.
cfg.TRAIN.MOMENTUM = 1.
# LR scheduler from 'CosineAnnealingLR', 'CosineLR', 'StepLR'
cfg.TRAIN.SCHEDULER = 'CosineAnnealingLR'
# number of warmup epochs when using CosineAnnealingLR or CosineLR
cfg.TRAIN.WARMUP = 0
# When using StepLR, number of epochs to decay learning rate
cfg.TRAIN.DECAY_RATE = 0.8
# max norm of the gradients, 0. to disable gradient clipping
cfg.TRAIN.GRAD_CLIPPING = 0.
# No. of steps of gradient accumulation, 1 means no accumulation
cfg.TRAIN.GRADIENT_ACC_STEPS = 1
# Use early stopping with a set patience
cfg.TRAIN.EARLY_STOPPING = True
cfg.TRAIN.PATIENCE = 10
# Using Sharpness-Aware Minimization (see https://github.com/davda54/sam)
cfg.TRAIN.SAM = False
# load pretrained model from timm image models
cfg.TRAIN.PRETRAINED = True
# continue training from existing model
cfg.TRAIN.CONTINUE = False

# -----------------------------------------------------------------------------

# Evaluate val and test set
cfg.EVALUATE_VAL = True
# Specify GPU device, e.g. 'cuda:0'
cfg.DEVICE = 'cuda'
