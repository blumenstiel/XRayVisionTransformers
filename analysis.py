
import numpy as np
import pandas as pd
from glob import glob

from src.utils import root
from src.plots import plot_roc_curve, save_roc_plot, label_dict, get_colors, plot_dataset_quality


def plot_roc_comparison(model_dict, path=None, zoom=False):
    """
    Plot the ROC curve for multiple models.
    """
    colors = get_colors(len(model_dict))

    for i, (model, dir) in enumerate(model_dict.items()):
        data = np.load(root / dir / 'roc_values.npz')
        tpr, fpr, auc = data.values()
        linestyle = '-' if 'vit' in model else '--'
        plot_roc_curve(fpr, tpr, label=model, color=colors[i], linestyle=linestyle)

    plot_dataset_quality()

    save_roc_plot(path=path, zoom=zoom)


def evaluation_table(models_dir, path=None):
    """
    Create a table comparing accuracy and AUROC of multiple models
    """
    # get all evaluation values
    results = pd.DataFrame()
    for file in glob(str(models_dir / '*' / 'evaluation.csv')):
        results = pd.concat([results, pd.read_csv(file, index_col=0)], axis=1)

    # switch axis, drop loss, change labels, round values
    results = results.T
    results = results[['test acc', 'test auroc']]
    results['test acc'] = (results['test acc'].astype(float) * 100).round(2)
    results['test auroc'] = (results['test auroc'].astype(float) * 100).round(2)
    results.rename(index=label_dict, columns={'test acc': 'Accuracy (%)', 'test auroc': 'AUROC (%)'}, inplace=True)

    # export results
    if path is not None:
        results.to_csv(path)
    else:
        print(results)


def evaluation_table_single_extremity(models_dir, path=None):
    """
    Create a table comparing the accuracy of multiple models by extremity
    """
    # get all evaluation values
    results = pd.DataFrame()
    for file in glob(str(models_dir / '*' / 'acc_per_extremity.csv')):
        results = pd.concat([results, pd.read_csv(file, index_col=0)], axis=1)

    # switch axis, drop loss, change labels, round values
    results = (results * 100).round(2)
    results.index = [e.title() for e in results.index]
    results.rename(columns=label_dict, inplace=True)
    results = results.T

    # export results
    if path is not None:
        results.to_csv(path)
    else:
        print(results)


def compare_trials(trials_dir, path=None):
    """
    Compare results of different models
    """
    # get all evaluation values
    results = pd.DataFrame()
    for file in glob(str(trials_dir / '*' / 'evaluation.csv')):
        df = pd.read_csv(file, index_col=0)
        df.columns = [file.split('/')[-2]]
        results = pd.concat([results, df], axis=1)

    results = results.T
    results['test acc'] = (results['test acc'].astype(float) * 100).round(2)
    results['test auroc'] = (results['test auroc'].astype(float) * 100).round(2)
    results['test loss'] = results['test loss'].astype(float).round(6)
    if 'val acc' in results.columns:
        results['val acc'] = (results['val acc'].astype(float) * 100).round(2)
        results['val auroc'] = (results['val auroc'].astype(float) * 100).round(2)
        results['val loss'] = results['val loss'].astype(float).round(6)
        results['acc'] = results['val acc'].astype(str) + ' / ' + results['test acc'].astype(str)

    # export results
    if path is not None:
        results.to_csv(path)
    else:
        print(results)


def compare_single_extremity(model_name, path=None):
    """
    Create a table to compare the results of single extremity models with the results of the multi-extremity model.
    """
    # get all evaluation values
    multi_results = pd.read_csv(root / 'results' / model_name / 'acc_per_extremity.csv', index_col=0)
    single_results = pd.DataFrame()
    for file in glob(str(root / 'trials/single_extremity' / f'{model_name}_*' / 'acc_per_extremity.csv')):
        single_results = pd.concat([single_results, pd.read_csv(file, index_col=0)])

    # combine results, rename index, round values
    results = pd.concat([multi_results, single_results], axis=1)
    print(results)
    results.columns = ['Combined model', 'Single model']
    results.index = [e.title() for e in results.index]
    results = (results * 100).round(2)

    # export results
    if path is not None:
        results.to_csv(path)
    else:
        print(results)


if __name__ == '__main__':
    analysis_dir = root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    # compare models for hyper parameter tuning
    compare_trials(root / 'trials' / 'optimizer', path=analysis_dir / 'optimizer_comparison.csv')
    compare_trials(root / 'trials' / 'scheduler', path=analysis_dir / 'scheduler_comparison.csv')
    compare_trials(root / 'trials' / 'augmentation_regularization', path=analysis_dir / 'aug_reg_comparison.csv')
    compare_trials(root / 'trials' / 'pooling', path=analysis_dir / 'pooling_comparison.csv')
    compare_trials(root / 'trials' / 'sam', path=analysis_dir / 'sam_comparison.csv')
    compare_trials(root / 'trials' / 'epochs', path=analysis_dir / 'epochs_comparison.csv')
    compare_trials(root / 'results', path=analysis_dir / 'results_comparison.csv')
    compare_trials(root / 'results_w_aug', path=analysis_dir / 'results_w_aug_comparison.csv')

    # create table with evaluation results
    evaluation_table(root / 'results', analysis_dir / 'evaluation_table.csv')
    evaluation_table(root / 'results_wo_aug', analysis_dir / 'evaluation_table_wo_aug.csv')
    evaluation_table_single_extremity(root / 'results', analysis_dir / 'evaluation_single_extremity.csv')

    # compare single extremity models with multi-extremity model
    compare_single_extremity('vit_small', analysis_dir / 'single_extremity_vit_small.csv')
    compare_single_extremity('resnet_50x1', analysis_dir / 'single_extremity_resnet_50x1.csv')

    # create roc plots
    model_dict = {
        'vit_small_patch16_384': 'results/vit_small',
        'vit_base_patch16_384': 'results/vit_base',
        'vit_large_patch16_384': 'results/vit_large',
        'vit_base_r50_s16_384': 'results/vit_base_r50',
        'resnetv2_50x1_bitm': 'results/resnet_50x1',
        'resnetv2_152x2_bitm': 'results/resnet_152x2',
    }
    plot_roc_comparison(model_dict, analysis_dir / 'roc_comparison.pdf')
    plot_roc_comparison(model_dict, analysis_dir / 'roc_comparison_zoomed.pdf', zoom=True)

