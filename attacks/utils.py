from torchvision.transforms import Compose, Normalize

from logger import logger


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        logger.warning(f"Failed in weights_init for {m._get_name()}.weight")

    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        logger.warning(f"Failed in weights_init for {m._get_name()}.bias")


def denormalize(img, mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]):
    trans = Compose(
        [
            Normalize(mean=[0.0, 0.0, 0.0], std=[1 / std[0], 1 / std[1], 1 / std[2]]),
            Normalize(mean=[-mean[0], -mean[1], -mean[2]], std=[1.0, 1.0, 1.0]),
        ]
    )
    return trans(img)
