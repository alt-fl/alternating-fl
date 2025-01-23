from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms


class Cinic10Data(Dataset):
    def __init__(self, data, targets, classes, transform):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.targets[index]

    def __len__(self):
        return len(self.data)


def open_as_img(p, dst):
    with dst.open(p) as f:
        img = Image.open(f).convert("RGB")
        img.load()
        return img


def read_dataset(cinic10_path, cifar10_path):
    cifar10 = CIFAR10(cifar10_path)
    classes = cifar10.classes
    cls_to_idx = cifar10.class_to_idx

    def get_target(path_str):
        for cls in classes:
            if cls + "/" in path_str:
                return cls_to_idx[cls]

        raise ValueError(f"{path_str!r} is not an image of CIFAR10 classes")

    with ZipFile(cinic10_path, "r") as cinic10_zip:
        train_data = []
        train_targets = []
        test_data = []
        test_targets = []

        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trans_cinic10 = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean, std=cinic_std),
            ]
        )
        for path in tqdm(cinic10_zip.namelist()):
            if path[-4:] == ".png":
                if "train/" in path:
                    train_data.append(open_as_img(path, cinic10_zip))
                    train_targets.append(get_target(path))
                elif "test/" in path:
                    test_data.append(open_as_img(path, cinic10_zip))
                    test_targets.append(get_target(path))

        trainset = Cinic10Data(train_data, train_targets, classes, trans_cinic10)
        testset = Cinic10Data(test_data, test_targets, classes, trans_cinic10)

        return trainset, testset
