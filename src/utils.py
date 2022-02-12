from pathlib import Path
import torch
import numpy as np
import argparse


def get_project_root() -> Path:
    # using pathlib to ensure that code works on all OS systems
    return Path(__file__).parent.parent


# init root
root = get_project_root()


def get_device(d):
    if torch.cuda.is_available():
        if d != 'cuda':
            device = torch.device(d)
            torch.cuda.set_device(device)
            return device
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_seed(seed):
    """
        Setting random seeds
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_parser():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument(
        "-config",
        help="path to config file",
        default="configs/vit_small.yaml",
        metavar="FILE",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def save_cfg(cfg, file):
    with open(file, 'w') as f:
        f.write('Config:\n')
        print(cfg, file=f)
        f.write('\n')


def get_hparams(cfg):
    return {
        'Model': cfg.MODEL.NAME,
        'Optimizer': cfg.TRAIN.OPTIMIZER,
        'Learning Rate': cfg.TRAIN.LEARNING_RATE,
        'Scheduler': cfg.TRAIN.SCHEDULER,
        'Warmup steps': cfg.TRAIN.WARMUP,
        'Decay rate': cfg.TRAIN.DECAY_RATE,
        'Gradient clipping': cfg.TRAIN.GRAD_CLIPPING,
        'Gradient accumulation': cfg.TRAIN.GRADIENT_ACC_STEPS,
        'Regularization': cfg.MODEL.REGULARIZATION,
        'Augmentation': cfg.DATASET.RAND_AUGMENT,
        'SAM': cfg.TRAIN.SAM,
        'Non-linear head': cfg.MODEL.NONLINEAR_HEAD,
        'Epochs': cfg.TRAIN.EPOCHS,
    }


def hparams_to_tensorboard(writer, params_dict, meter_values):
    """Write config with test results to tensorboard"""
    metric_dict = {
        'test/Loss': meter_values['test loss'],
        'test/Accuracy': meter_values['test acc'],
        'test/AUROC': meter_values['test auroc'],
    }
    writer.add_hparams(params_dict, metric_dict, run_name='test')


def print_file(s, file):
    print(s)
    with open(file, 'a') as f:
        print(s, file=f)


def extract_meter_values(meters):
    """save meter values to a dict"""
    ret = {}

    for split in meters.keys():
        for field, meter in meters[split].items():
            if field == 'auroc':
                if len(meter.scores) > 0:
                    ret[f'{split} {field}'] = meter.value()[0]
                else:
                    # if no samples were seen, return nan
                    ret[f'{split} {field}'] = np.nan
            elif field == 'acc_per_extremity':
                pass
            elif field == 'time':
                ret[f'{split} {field}'] = meter.value() / 60
            else:
                ret[f'{split} {field}'] = meter.mean

    return ret


def render_meter_values(meter_values):
    """Create a string representation of the meter values"""
    field_info = []
    for field, val in meter_values.items():
        field_info.append(f"{field} = {val:0.4f}")
    return ', '.join(field_info)


def meter_values_to_tensorboard(writer, meter_values, epoch):
    """Write meter values to tensorboard"""
    meter_dict = {
        'train loss': 'Loss/train',
        'train acc': 'Accuracy/train',
        'train auroc': 'AUROC/train',
        'train time': 'Time/train',
        'val loss': 'Loss/val',
        'val acc': 'Accuracy/val',
        'val auroc': 'AUROC/val',
        'val time': 'Time/train',
        'test loss': 'Loss/test',
        'test acc': 'Accuracy/test',
        'test auroc': 'AUROC/test',
        'test time': 'Time/train',
    }
    for field, val in meter_values.items():
        writer.add_scalar(meter_dict[field], val, epoch)


def roc_to_tensorboard(writer, tpr, fpr):
    """Write ROC curve to tensorboard"""
    for tp, fp in zip(tpr, fpr):
        writer.add_scalar('ROC curve/test', int(tp * 100), fp * 100)
    # add initial point, so the ROC curve starts at (0, 0)
    writer.add_scalar('ROC curve/test', 0, 0.)


class ExtremityAccMeter():
    """
    Meter for calculating the Accuracy for every extremity separately.
    Functionality is similar to AverageValueMeter class from torchnet.
    """
    def __init__(self, num_extremities):
        self.reset()
        self.num_extremities = num_extremities

    def reset(self):
        self.acc = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.count = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, acc, count):
        if torch.is_tensor(acc):
            acc = acc.cpu().squeeze().numpy()
        if torch.is_tensor(count):
            count = count.cpu().squeeze().numpy()

        self.acc = np.append(self.acc, acc)
        self.count = np.append(self.count, count)

    def value(self):
        # reshape to batch results
        acc_per_extremity = np.nan_to_num(self.acc.reshape(-1, self.num_extremities))
        count = self.count.reshape(-1, self.num_extremities) + 1e-10

        acc_per_extremity = np.average(acc_per_extremity, axis=0, weights=count)
        return acc_per_extremity
