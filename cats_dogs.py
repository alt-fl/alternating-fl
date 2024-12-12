from zipfile import ZipFile
from random import sample, shuffle
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


class CatsDogsData(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]

    def __len__(self):
        return len(self.data)


def get_mean_std(loader):
    # Compute the mean and standard deviation of the images for each channel
    # in the dataset
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for images, _ in loader:
        mean += images.mean(axis=(0, 2, 3))
        std += images.std(axis=(0, 2, 3))

    mean /= len(loader)
    std /= len(loader)

    return mean, std


def load_dst(zip_path):
    """
    Very strict assumption about the zip file, it needs to looks like
    train/
      |-img1
      |-img2
      |-...
      .
      .
      .
    """

    def open_as_img(p, dst):
        with dst.open(p) as f:
            img = Image.open(f)
            img.load()
            return img

    with ZipFile(zip_path, "r") as dst_zip:
        cat_label = 3
        dog_label = 5

        imgs = dst_zip.namelist()[1:]
        dog_imgs = sample(list(filter(lambda x: "dog" in x, imgs)), 5000)
        cat_imgs = sample(list(filter(lambda x: "cat" in x, imgs)), 5000)

        dog_imgs = list(map(lambda p: (open_as_img(p, dst_zip), dog_label), dog_imgs))
        cat_imgs = list(map(lambda p: (open_as_img(p, dst_zip), cat_label), cat_imgs))

        all_imgs = []
        all_imgs.extend(dog_imgs)
        all_imgs.extend(cat_imgs)

        shuffle(all_imgs)

        data, labels = zip(*all_imgs)

        transf = Compose([Resize((32, 32)), ToTensor()])
        dst = DataLoader(CatsDogsData(data, labels, transform=transf), batch_size=64)

        mean, std = get_mean_std(dst)

        transf = Compose([Resize((32, 32)), ToTensor(), Normalize(mean=mean, std=std)])
        dst = CatsDogsData(data, labels, transform=transf)
        return dst, labels
