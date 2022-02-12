
import numpy as np
import pandas as pd
import torch
import torchnet as tnt
from torch.utils.tensorboard import SummaryWriter
from timm import optim
from timm.scheduler import CosineLRScheduler, StepLRScheduler
from types import SimpleNamespace
from tqdm import tqdm

from src import evaluation
from src.sam import SAM
from src.model import Classifier
from src.dataset import get_data_loaders
from src.utils import root, get_device, save_cfg, print_file, extract_meter_values, render_meter_values, \
    meter_values_to_tensorboard, ExtremityAccMeter


def train(cfg, do_evaluation=False):
    """
    Train the model.
    """
    # Save config to trace file
    trace_file = root / cfg.MODEL.DIR / f'train.log'
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    save_cfg(cfg, trace_file)

    # data loaders
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # set device
    device = get_device(cfg.DEVICE)

    # initialize model
    model = Classifier(cfg)
    model.to(device)

    # init optimizer
    lr_scaled = cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.BATCH_SIZE / 512
    if cfg.TRAIN.SAM:
        # Using Sharpness-Aware Minimization
        # see https://github.com/davda54/sam for more information
        base_optim = getattr(torch.optim, cfg.TRAIN.OPTIMIZER)
        if 'Adam' in cfg.TRAIN.OPTIMIZER:
            optimizer = SAM(model.parameters(), base_optim, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                            lr=lr_scaled, adaptive=False)
        elif 'SGD' in cfg.TRAIN.OPTIMIZER:
            optimizer = SAM(model.parameters(), torch.optim.SGD, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                            lr=lr_scaled, momentum=cfg.TRAIN.MOMENTUM, adaptive=True)
        else:
            raise ValueError(f'Optimizer not implemented for SAM: {cfg.TRAIN.OPTIMIZER}')
    else:
        args = SimpleNamespace()
        args.weight_decay = cfg.TRAIN.WEIGHT_DECAY
        args.lr = lr_scaled
        args.opt = cfg.TRAIN.OPTIMIZER
        args.momentum = cfg.TRAIN.MOMENTUM
        optimizer = optim.create_optimizer(args, model)

    # init lr scheduler
    if cfg.TRAIN.SCHEDULER == 'StepLR':
        # Use step-wise learning rate decay
        scheduler = StepLRScheduler(optimizer,
                                    decay_t=len(train_loader),
                                    decay_rate=cfg.TRAIN.DECAY_RATE)
    elif cfg.TRAIN.SCHEDULER == 'CosineLR':
        # Use cosine learning rate decay
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=cfg.TRAIN.EPOCHS * len(train_loader),
                                      warmup_t=cfg.TRAIN.WARMUP,
                                      cycle_limit=1,
                                      warmup_prefix=True)
    elif cfg.TRAIN.SCHEDULER == 'CosineAnnealingLR':
        # Use cosine annealing learning rate
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=10 * len(train_loader),
                                      warmup_t=cfg.TRAIN.WARMUP,
                                      cycle_limit=0,
                                      warmup_prefix=True)
    else:
        raise ValueError(f'Unknown scheduler: {cfg.TRAIN.SCHEDULER}')

    # initialize meters
    meters = {split: {'loss': tnt.meter.AverageValueMeter(),
                      'acc': tnt.meter.AverageValueMeter(),
                      'auroc': tnt.meter.AUCMeter(),
                      'acc_per_extremity': ExtremityAccMeter(len(cfg.DATASET.EXTREMITIES)),
                      'time': tnt.meter.TimeMeter(unit=False),
                       } for split in ['train', 'val']}
    meters_results = pd.DataFrame()
    meters_results.index.name = 'epoch'

    # initialize tensorboard writer
    tensorboard_writer = SummaryWriter(log_dir=f'./tensorboard/{cfg.MODEL.DIR}')
    tensorboard_writer.add_text('Config', str(cfg))

    # initialize early stopping
    patience_counter = 0
    patience = cfg.TRAIN.PATIENCE if cfg.TRAIN.EARLY_STOPPING else cfg.TRAIN.EPOCHS

    start_epoch = 1
    global_step = 0
    # continue training from best model
    if cfg.TRAIN.CONTINUE and model.path.is_file() and trace_file.with_suffix('.csv').is_file():
        # load previous results and update settings.
        # As only the best model is saved, training is continued from this checkpoint
        model.load_weights()
        meters_results = pd.read_csv(trace_file.with_suffix('.csv'), index_col=0)
        start_epoch = meters_results['val loss'].idxmin() + 1
        assert len(meters_results) - start_epoch + 1 >= patience, 'Training was stopped because of early stopping.'
        # drop meter_results after best epoch
        meters_results = meters_results.iloc[:start_epoch]
        # update lr
        global_step = len(train_loader) * (start_epoch - 1)
        scheduler.step(global_step)
        print_file(f'Continuing training from epoch {start_epoch}', trace_file)

    # train
    print_file(f'Train model {cfg.MODEL.DIR} for {cfg.TRAIN.EPOCHS} epochs', trace_file)
    best_loss = np.inf
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS + 1):
        # reset meters
        for set, set_meters in meters.items():
            for field, meter in set_meters.items():
                meter.reset()

        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch:d} train'):
            if torch.cuda.is_available():
                # write tensors to GPU
                for k, t in batch.items():
                    batch[k] = t.cuda(non_blocking=True)

            # forward pass
            loss, output = model.loss(batch)

            if cfg.TRAIN.SAM:
                # Using Sharpness-Aware Minimization
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # second forward-backward pass
                loss, output = model.loss(batch)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # standard backward step
                loss = loss / cfg.TRAIN.GRADIENT_ACC_STEPS
                loss.backward()
                if cfg.TRAIN.GRAD_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIPPING)
                if (global_step % cfg.TRAIN.GRADIENT_ACC_STEPS) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # update learning rate
            global_step += 1
            scheduler.step(global_step)

            # add results to meters
            meters['train']['loss'].add(output['loss'])
            meters['train']['acc'].add(output['acc'])
            meters['train']['auroc'].add(output['p_y_pos'], batch['labels'].cpu().numpy())

        # evaluate model
        meters['val']['time'].reset()
        evaluation.evaluate(model,
                            val_loader,
                            meters['val'],
                            desc=f'Epoch {epoch:d} valid')

        # print validation results
        meter_vals = extract_meter_values(meters)
        print_file(f'Epoch {epoch:02d}: {render_meter_values(meter_vals)}', trace_file)

        # save results for tensorboard
        meter_values_to_tensorboard(tensorboard_writer, meter_vals, epoch)

        # save results as csv
        meters_results = pd.concat([meters_results, pd.DataFrame(meter_vals, index=[epoch])])
        meters_results.to_csv(trace_file.with_suffix('.csv'))

        if meter_vals['val loss'] < best_loss:
            # save model
            best_loss = meter_vals['val loss']
            print_file(f'==> best model (loss = {best_loss:0.6f}), saving model...', trace_file)

            model.cpu()
            torch.save(model.state_dict(), model.path)
            model.to(device)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print_file(f'Early stopping at epoch {epoch:d}', trace_file)
            break

    print_file('Finished training.', trace_file)
    tensorboard_writer.close()

    if do_evaluation:
        evaluation.run_evaluation(cfg)
