import numpy as np
import cv2 as cv
import pandas as pd
import torch

from numpy import ndarray
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional, Callable, Tuple


class Mnist(Dataset):
    """MNIST parsing dataset."""

    def __init__(
            self,
            path: Union[Path, str],
            config: object,
            transform: Optional[Callable] = None
    ):
        """
        Constuctor for MNIST dataset
        :param path: path to csv file;
        :param config: main configurate class;
        :param transform: data augmentation augmentation;
        """

        super().__init__()
        # load csv file
        self.csv_file = pd.read_csv(path, names=['ID', 'PATH', 'CLASS_ID'])
        self.transform = transform
        self.config = config

    def __len__(self):
        """Return dataset length."""
        return len(self.csv_file)

    def __getitem__(self, idx: int) -> \
            Tuple[Union[ndarray, torch.Tensor], int]:
        """
        Get data sample
        :param idx: id data for load
        :return:
                image: Union[ndarray, torch.Tensor]
                label: int
        """
        image = cv.imread(self.csv_file['PATH'][idx], cv.IMREAD_UNCHANGED)
        image = np.expand_dims(image, 2)
        label = self.csv_file['CLASS_ID'][idx]
        if self.transform is not None:
            image, label = self.transform(image=image, label=label)

        return image, label