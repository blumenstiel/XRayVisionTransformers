
import pandas as pd
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, CenterCrop, \
    ToTensor
from timm.data import RandAugment, rand_augment_ops
from src.utils import root

extremity_dict = {
    'ELBOW': 0,
    'FINGER': 1,
    'FOREARM': 2,
    'HAND': 3,
    'HUMERUS': 4,
    'SHOULDER': 5,
    'WRIST': 6
}

class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = Compose([])

    def __len__(self):
        return len(self.df)

    def set_transform(self, transform):
        self.transform = transform

    def get_study(self, idx):
        if isinstance(idx, list):
            return self.get_batch(idx)
        images = []
        for image_path in self.df.images.iloc[idx]:
            image = pil_loader(root / image_path)
            image = self.transform(image)
            images.append(image)
        label = self.df.label.iloc[idx]
        count = len(images)
        extremity = self.df.extremity.iloc[idx]
        sample = {'images': images,
                  'label': label,
                  'count': count,
                  'extremity': extremity}
        return sample

    def __getitem__(self, idx):
        if isinstance(idx, int):
            sample = self.get_study(idx)
            return sample
        else:
            samples = []
            for i in idx:
                samples.append(self.__getitem__(i))
            return samples


def load_MURA(extremities=None):  # 'SHOULDER'
    # function for extracting relevant information for MURA dataset
    def process_MURA(path):
        ds = pd.read_csv(path, header=None)
        ds.columns = ['path', 'label']
        # get extremities, filter them and replace with categorical number
        ds['extremity'] = ds['path'].apply(lambda x: x.split('/')[2][3:])
        # get patient id and images paths
        ds['patient'] = ds['path'].apply(lambda x: x.split('/')[3][7:])
        ds['images'] = ds['path'].apply(lambda x: glob(x + '*.png'))
        return ds

    # get train and val dataset
    path = root / 'MURA-v1.1'
    train_ds = process_MURA(path / 'train_labeled_studies.csv')
    val_ds = process_MURA(path / 'valid_labeled_studies.csv')

    # split up train_ds into train and test set
    test_ds = train_ds.sample(frac=0.1, random_state=42)
    train_ds.drop(test_ds.index, inplace=True)

    if extremities is None:
        extremities = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']
    # filter extremities and replace with categorical number
    train_ds = train_ds[train_ds['extremity'].isin(extremities)]
    train_ds['extremity'] = train_ds['extremity'].map(extremity_dict)
    val_ds = val_ds[val_ds['extremity'].isin(extremities)]
    val_ds['extremity'] = val_ds['extremity'].map(extremity_dict)
    test_ds = test_ds[test_ds['extremity'].isin(extremities)]
    test_ds['extremity'] = test_ds['extremity'].map(extremity_dict)

    # init datasets
    train_ds = ImageDataset(train_ds)
    val_ds = ImageDataset(val_ds)
    test_ds = ImageDataset(test_ds)

    return train_ds, val_ds, test_ds


def get_transforms(model_size, mean, std, rand_augment):
    # values for RandAugment
    rand_augment_dict = {
        'none': (0, 0),
        'light': (2, 10),
        'medium': (2, 15),
        'heavy': (4, 20),
    }
    num_layers, magnitude = rand_augment_dict[rand_augment]

    train_transforms = Compose([
        RandomResizedCrop(model_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        RandomHorizontalFlip(),
        RandAugment(ops=rand_augment_ops(magnitude=magnitude), num_layers=num_layers),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    test_transforms = Compose([
        Resize(model_size),
        CenterCrop(model_size),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    return train_transforms, test_transforms


def get_data_loaders(cfg):
    # Load dataset
    train_ds, val_ds, test_ds = load_MURA(cfg.DATASET.EXTREMITIES)

    # Set the transforms
    train_transforms, test_transforms = get_transforms(cfg.MODEL.INPUT_SIZE,
                                                       cfg.MODEL.MEAN,
                                                       cfg.MODEL.STD,
                                                       cfg.DATASET.RAND_AUGMENT)
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(test_transforms)
    test_ds.set_transform(test_transforms)

    # collate function for dataloader
    def collate_fn(examples):
        """
        Stack all images in examples. 'study_id' contains the id of the study in the batch.
        """
        images = torch.concat([torch.stack(example['images']) for example in examples])
        labels = torch.tensor([example['label'] for example in examples])
        study = torch.concat([torch.Tensor([i] * x['count']) for i, x in enumerate(examples)]).long()
        extremities = torch.tensor([example['extremity'] for example in examples])
        return {'images': images, 'labels': labels, 'study_ids': study, 'extremities': extremities}

    # collate function for train dataloader
    def train_collate_fn(examples):
        """
        Stack images in examples. Uses always two random images per study for training to avoid memory issues.
        'study_id' contains the id of the study in the batch.
        """
        images = []
        for example in examples:
            random_idx = torch.randint(0, example['count'], (2,))
            images.append(torch.stack([example['images'][i] for i in random_idx]))
        images = torch.concat(images)
        labels = torch.tensor([example['label'] for example in examples])
        study_ids = torch.arange(len(examples)).repeat_interleave(2).long()
        extremities = torch.tensor([example['extremity'] for example in examples])
        return {'images': images, 'labels': labels, 'study_ids': study_ids, 'extremities': extremities}

    if cfg.DATASET.SINGLE_VIEW:
        def collate_fn_single(examples):
            """
            Stack images in examples. Uses always two random images per study for training to avoid memory issues.
            'study_id' contains the id of the study in the batch.
            """
            images = torch.concat([torch.stack([example['images'][0]]) for example in examples])
            labels = torch.tensor([example['label'] for example in examples])
            study_ids = torch.arange(len(examples)).long()
            extremities = torch.tensor([example['extremity'] for example in examples])
            return {'images': images, 'labels': labels, 'study_ids': study_ids, 'extremities': extremities}
        collate_fn = collate_fn_single

    # create data loaders
    n_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=train_collate_fn, batch_size=cfg.TRAIN.BATCH_SIZE,
                              drop_last=True, pin_memory=True, num_workers=n_workers)
    val_loader = DataLoader(val_ds, shuffle=True, collate_fn=collate_fn, batch_size=cfg.TRAIN.BATCH_SIZE,
                            drop_last=False, pin_memory=True, num_workers=n_workers)
    test_loader = DataLoader(test_ds, shuffle=True, collate_fn=collate_fn, batch_size=cfg.TRAIN.BATCH_SIZE,
                             drop_last=False, pin_memory=True, num_workers=n_workers)

    return train_loader, val_loader, test_loader
