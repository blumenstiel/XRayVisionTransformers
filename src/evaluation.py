
import torch
import pandas as pd
import numpy as np
import torchnet as tnt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.model import Classifier
from src.dataset import get_data_loaders
from src.plots import plot_roc_curve, save_roc_plot
from src.utils import root, get_device, save_cfg, print_file, extract_meter_values, render_meter_values, \
    hparams_to_tensorboard, roc_to_tensorboard, get_hparams, ExtremityAccMeter


def evaluate(model, loader, meters, desc='Evaluation'):
    """
    Evaluate model on given data loader.
    """
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if torch.cuda.is_available():
                # write tensors to GPU
                for k, t in batch.items():
                    batch[k] = t.cuda(non_blocking=True)

            # get metrics
            _, output = model.loss(batch)

            # add results to meters
            meters['loss'].add(output['loss'])
            meters['acc'].add(output['acc'])
            meters['auroc'].add(output['p_y_pos'], batch['labels'].cpu().numpy())
            meters['acc_per_extremity'].add(output['acc_per_extremity'],
                                            torch.bincount(batch['extremities'],
                                                           minlength=len(output['acc_per_extremity'])))


def run_evaluation(cfg):
    """
    Run evaluation based on cfg.
    """

    # init trace file
    output_dir = root / cfg.MODEL.DIR
    if cfg.DATASET.SINGLE_VIEW:
        output_dir = output_dir / 'single_view'
    trace_file = output_dir / f'evaluation.log'
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    save_cfg(cfg, trace_file)

    # set device
    device = get_device(cfg.DEVICE)

    # init meters
    if cfg.EVALUATE_VAL:
        splits = ['val', 'test']
    else:
        splits = ['test']
    meters = {split: {'loss': tnt.meter.AverageValueMeter(),
                      'acc': tnt.meter.AverageValueMeter(),
                      'auroc': tnt.meter.AUCMeter(),
                      'acc_per_extremity': ExtremityAccMeter(len(cfg.DATASET.EXTREMITIES)),
                      } for split in splits}
    tensorboard_writer = SummaryWriter(log_dir=f'./tensorboard/{cfg.MODEL.DIR}')

    # load model
    model = Classifier(cfg, load_weights=True)
    model.to(device)
    
    # get data loader
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # evaluate
    print('Evaluating...')
    if cfg.EVALUATE_VAL:
        evaluate(model, val_loader, meters['val'], desc='Evaluate val set')
    evaluate(model, test_loader, meters['test'], desc='Evaluate test set')
            
    # print results in trace file
    meter_vals = extract_meter_values(meters)
    print_file(f'Evaluation: {render_meter_values(meter_vals)}', trace_file)

    # save results as csv
    params_dict = get_hparams(cfg)
    meters_results = pd.Series(dict(**meter_vals, **params_dict), name=cfg.MODEL.NAME)
    meters_results.to_csv(trace_file.with_suffix('.csv'))

    # save acc per extremity
    acc_per_extremity = meters['test']['acc_per_extremity'].value()
    acc_per_extremity = pd.Series(acc_per_extremity, index=cfg.DATASET.EXTREMITIES, name=cfg.MODEL.NAME)
    acc_per_extremity.to_csv(output_dir / f'acc_per_extremity.csv')

    # plot ROC curve
    auc, tpr, fpr = meters['test']['auroc'].value()
    plot_roc_curve(fpr, tpr, auc, cfg.MODEL.NAME)
    save_roc_plot(output_dir / f'roc_curve.png')
    # save ROC values
    np.savez(output_dir / 'roc_values.npz', tpr=tpr, fpr=fpr, auc=auc)

    # save values and plots to tensorboard
    hparams_to_tensorboard(tensorboard_writer, params_dict, meter_vals)
    roc_to_tensorboard(tensorboard_writer, tpr, fpr)
    tensorboard_writer.add_pr_curve('PR curve', meters['test']['auroc'].targets, meters['test']['auroc'].scores)
    tensorboard_writer.close()

