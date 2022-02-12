import torch

from src.config import cfg
from src.utils import get_parser, set_seed, root, get_device
from src.dataset import load_MURA, extremity_dict
from src.model import Classifier
from src.plots import plot_attention_map
from src.dataset import get_transforms


def main():
    # Parse arguments
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Init seed for reproducibility
    set_seed(1234)

    device = get_device(cfg.DEVICE)

    _, _, test_ds = load_MURA()

    # Load model
    model = Classifier(cfg, load_weights=True)
    model.to(device)
    model.eval()

    # get transformations
    _, transform = get_transforms(cfg.MODEL.INPUT_SIZE,
                                  cfg.MODEL.MEAN,
                                  cfg.MODEL.STD,
                                  cfg.DATASET.RAND_AUGMENT)


    dir_path = root / 'analysis' / 'attention_maps' / cfg.MODEL.DIR.split('/')[-1]
    dir_path.mkdir(exist_ok=True, parents=True)

    # reverse extremity_dict
    extremity_dict_reversed = dict(map(reversed, extremity_dict.items()))

    # save attention maps for the first 100 studies in the test set
    with torch.no_grad():
        for idx in range(100):
            instance = test_ds[idx]
            extremity = extremity_dict_reversed[instance["extremity"]]
            if extremity not in cfg.DATASET.EXTREMITIES:
                continue
            label = 'positive' if instance["label"] == 1 else 'negative'
            for i, image in enumerate(instance['images']):
                # transform image
                input = transform(image).to(device)

                # get and save attention map
                attention_map = model.get_attention_map(input)
                plot_attention_map(image, attention_map, path=dir_path / f'{idx}_{i}_{extremity}_{label}.png')


if __name__ == '__main__':
    main()
