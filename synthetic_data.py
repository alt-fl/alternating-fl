from collections.abc import Iterator
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class InterleavingRounds(Iterator):
    """
    Simple iterator that, given an interleaving ratio, returns the current round
    number and if it is an authentic round.

    The interleaving ratio is a tuple of non-negative integers denoting the
    frequencies of authentic and synthetic rounds. I.e., (AR, SR) means that
    it will have AR rounds of authentic rounds followed by SR rounds of
    synthetic rounds, and so on.
    """

    def __init__(self, rounds=10, ratio=(1, 0)) -> None:
        auth, syn = ratio
        self.rounds = iter(range(rounds))
        self.rate = auth + syn
        self.auth = range(0, auth)
        self.syn = range(auth, auth + syn)

    def __next__(self):
        round = next(self.rounds)
        mod = round % self.rate
        is_auth = mod in self.auth

        # sanity check that it must be either authentic or synthetic rounds
        is_syn = mod in self.syn
        assert is_auth != is_syn

        return round, is_auth


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, num_classes=10):
        super().__init__()
        self.dataset = dataset
        self.transform = self.dataset.transform
        self.indices = np.array(indices)
        self.targets = np.array(dataset.targets)[indices]
        if isinstance(dataset, ImageFolder):
            self.data = [dataset.__getitem__(idx) for idx in tqdm(indices)]
        else:
            self.data = list(zip(dataset.data[indices], self.targets))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample, target = self.data[idx]
        if self.transform and not isinstance(self.dataset, ImageFolder):
            sample = self.transform(sample)
        return sample, target


class SyntheticCIFAR10(Dataset):
    def __init__(
        self,
        num_samples=50000,
        num_classes=10,
        image_shape=(3, 32, 32),
        train=True,
        transform=None,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.train = train
        self.transform = transform

        # Generate synthetic data and labels
        self.data = self._generate_data()
        self.targets = self._generate_labels()

        # Define class names
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def _generate_data(self):
        # Generate synthetic images (random pixels)
        data = np.random.randint(
            0, 256, size=(self.num_samples, *self.image_shape), dtype=np.uint8
        )
        return data

    def _generate_labels(self):
        # Generate synthetic labels (random integers representing class indices)
        labels = np.random.randint(0, self.num_classes, size=self.num_samples)
        return labels.tolist()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return image and its corresponding label
        image = torch.tensor(self.data[idx]).float() / 255.0  # Normalize to [0, 1]
        label = self.targets[idx]
        return image, label
