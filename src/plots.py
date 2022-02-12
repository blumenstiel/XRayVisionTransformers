
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    # "font.family": "Helvetica",
    'font.size': 14,
    'legend.fontsize': 14
    })

label_dict = {
    'vit_small_patch16_384': 'ViT-S/16',
    'vit_base_patch16_384': 'ViT-B/16',
    'vit_large_patch16_384': 'ViT-L/16',
    'vit_base_patch32_384': 'ViT-B/32',
    'vit_base_patch16_224': 'ViT-B/16-224',
    'vit_base_r50_s16_384': 'ViT-B/R50-S16',
    'deit_base_patch16_384': 'DeiT-B/16',
    'swin_base_patch4_window12_384': 'Swin-B/4-12',
    'resnetv2_50x1_bitm': 'ResNetV2-50x1',
    'resnetv2_101x1_bitm': 'ResNetV2-101x1',
    'resnetv2_50x3_bitm': 'ResNetV2-50x3',
    'resnetv2_152x2_bitm': 'ResNetV2-152x2',
    'tf_efficientnetv2_m_in21ft1k': 'EfficientNetV2-M',
}


global counter
counter = 0


def get_colors(n):
    return plt.cm.viridis(np.linspace(0, 0.95, n))


def plot_roc_curve(fpr, tpr, label=None, color=None, linestyle='-'):
    if color is None:
        global counter
        color = get_colors(8)[counter % 8]
        counter += 1

    label = label_dict[label] if label in label_dict else label
    plt.plot(fpr, tpr, label=f'{label}', color=color, linestyle=linestyle)


def plot_radiologist_values():
    # performance of radiologists reported in MURA Paper (https://arxiv.org/abs/1712.06957)
    # values for original test set, not the test set used in this project
    radiologist_values = [[0.095, 0.871], [0.171, 0.952], [0.203, 0.939]]
    plt.plot(*np.array(radiologist_values).T, 's', color='gray', label='Radiologist')


def plot_dataset_quality():
    # quality of MURA dataset reported in
    # Oakden-Rayner, L., “Exploring Large-scale Public Medical Image Datasets”, Academic Radiology 27(1), 2020
    sensitivity = 0.8
    specificity = 0.75
    plt.plot(1-specificity, sensitivity, 's', color='black', label='Dataset quality')


def save_roc_plot(path=None, draw_middle_line=False, zoom=False):
    if draw_middle_line:
        plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if zoom:
        plt.xlim([0.1, 0.3])
        plt.ylim([0.7, 0.9])
    else:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend(loc='lower right')
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_attention_map(original_img, att_map, plot_mask=True, path=None, title=False):
    # convert mask
    if plot_mask:
        att_map = np.array(Image.fromarray(att_map / att_map.max() * 255).resize(original_img.size, Image.BOX))
    if not plot_mask:
        att_map = Image.fromarray(att_map / att_map.max()).resize(original_img.size)
        att_map = (np.array(original_img) * np.array(att_map)[..., np.newaxis]).astype(int)

    # plot image and mask
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)

    # format
    if title:
        ax1.set_title('Original')
        if isinstance(title, str):
            ax2.set_title(title)
        else:
            ax2.set_title('Attention Map')
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()

    # export
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        plt.close()
