import numpy as np
import cv2 as cv
import torch
from typing import List


class Compose(object):
    """ Composes augmentation together. """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, label: int):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class ImageToFloat(object):
    """ Image to float32. """
    def __call__(self, image: np.ndarray, label: int):
        return image.astype(np.float32) / 255., label


class Resize(object):
    """ Resize image. Squad form only """
    def __init__(self, size: int):

        self.size = size

    def __call__(self, image: np.ndarray, label: int):

        # image shape
        h, w, c = image.shape

        width, height = self.size, self.size
        if h != height and w != width:
            image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
            image = np.expand_dims(image, 2)

        return image, label


class ToTorchTensor(object):
    """ Convert numpy array to torch tensor. """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, image: np.ndarray, label: int):
        """
        :param image: [H, W, C]
        :param label: int
        :return:
        """

        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32, device=self.device)

        return image, label


class RandomContrast(object):
    """ Change contrast value."""
    def __init__(self, lower=0.1, upper=1.):
        """
        :param lower: lower limit value;
        :param upper: upper limit value.
        """

        self.lower = lower
        self.upper = upper

        # validation of input values
        assert self.upper >= self.lower, "upper must be > lower."
        assert self.lower >= 0,          "lower must be non-negative."

    def __call__(self, image: np.ndarray, label: int):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha

        return image, label


class RandomBrightness(object):
    """ Change brightness value. """
    def __init__(self, delta=0.35):
        """
        :param delta: range to change brightness value.
        """
        # validation of input values
        assert delta >= 0.0
        assert delta <= 1.0
        self.delta = delta

    def __call__(self, image: np.ndarray, label: int):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta

        return image, label


class TrainTransforms(object):
    """ Train image transform."""
    def __init__(self):
        self.aug = Compose([
            ImageToFloat(),
            RandomContrast(),
            RandomBrightness(),
            ToTorchTensor()
        ])

    def __call__(self, image: np.ndarray, label: int):
        return self.aug(image, label)


class TestingTransform(object):
    """Testing image transform."""
    def __init__(self):
        self.aug = Compose([
            ImageToFloat(),
            ToTorchTensor()
        ])

    def __call__(self, image: np.ndarray, label: int):
        return self.aug(image, label)
