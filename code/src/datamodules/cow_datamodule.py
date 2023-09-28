from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from timm.data import resolve_data_config, create_transform, auto_augment_transform
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.model_selection import KFold


import os
import pandas as pd
from torchvision.io import read_image
import albumentations as A



class CowCustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.len = len(self.img_labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = cv2.imread(img_path)
        label = int(self.img_labels.iloc[idx, 2])
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CowDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            annotations_file: str = "data/annotations.csv",
            k: int = 1,  # fold number
            n_splits: int = 5,  # number of folds
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        p = 0.5
        # data transformations
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=p),
                A.VerticalFlip(p=p),
                A.ShiftScaleRotate(rotate_limit=0, p=p),
                A.RandomResizedCrop(height=512, width=512, p=p),
                A.MotionBlur(p=p),
                A.RandomBrightnessContrast(p=p),
                A.GaussNoise(var_limit=(0.1, 1), p=0.5),
                A.Rotate(limit=90, p=p),
                # transforms.Resize((512,512)),
                # transforms.ToTensor(),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(),

            ]
        )

        self.target_transforms = transforms.Compose(
            [
                # transforms.ToTensor(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            # testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = CowCustomImageDataset(self.hparams.annotations_file, self.hparams.data_dir,
                                            transform=self.transforms,target_transform=self.target_transforms)
            # if k == -1 disable kfold
            if self.hparams.k == -1:
                self.data_train, self.data_val = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(42),
                )

            else:
                kf = KFold(n_splits=self.hparams.n_splits, shuffle=True, random_state=42)
                all_splits = [k for k in kf.split(dataset)]
                train_indexes, val_indexes = all_splits[self.hparams.k]
                train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
                # self.data_train, self.data_val = dataset[train_indexes], dataset[val_indexes]

                self.data_train = torch.utils.data.Subset(dataset, train_indexes)

                self.data_val = torch.utils.data.Subset(dataset, val_indexes)


            self.data_test = self.data_val

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
