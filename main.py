from datetime import date
import os
from pathlib import Path

from opacus.data_loader import logging
import psutil
import time
import tracemalloc

import torch
import random
import numpy as np

from server import Server
from exp_wrapper import get_wrapper
from exp_args import ExperimentArgument

from logger import logger, configure_logger, MessageContentFilter


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_num_samples_per_class(data, dict_users, num_classes=10):
    summed = [0] * len(dict_users)
    for k in dict_users:
        counts = [0] * num_classes
        for id in dict_users[k]:
            counts[data[id][1]] += 1
        logger.debug(f"Client {k} samples per class: {counts}")
        summed[k] = sum(counts)
    logger.debug(f"Total samples/samples per client: {sum(summed)}, {summed}\n")


def main():
    args = ExperimentArgument()
    wrapper = get_wrapper()
    seed_torch(args.seed)

    log_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }.get(args.log_level) or logging.NOTSET
    configure_logger(level=log_level)

    logger.info(f"Client setting: {int(args.C * args.K)}/{args.K} active clients")

    today = str(date.today())
    filename = wrapper.get_output()
    output_dir = Path(".", today)
    output_path = Path(output_dir, filename)

    if not output_dir.exists():
        logger.info(f"Creating output directory {output_dir}")
        output_dir.mkdir(parents=True)
    else:
        logger.info(f"Output directory {output_dir} already exists, proceeding...")

    # split dataset into training (authentic and synthetic) and testing
    auth_data, syn_data, test_data = wrapper.get_data_split()
    # perform partition, and get the dictionary specifying the partition of
    # data that belongs to each  clients
    auth_dict_users, syn_dict_users = wrapper.partition_data()
    # create model
    model = wrapper.get_model()
    logger.info(f"Model: {args.model}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params}\n")

    logger.debug("==========authentic data==========")
    log_num_samples_per_class(auth_data, auth_dict_users, num_classes=args.num_classes)

    logger.debug("==========synthetic data==========")
    log_num_samples_per_class(syn_data, syn_dict_users, num_classes=args.num_classes)

    server = Server(
        model,
        wrapper.get_data(),
        auth_data,
        auth_dict_users,
        syn_data,
        syn_dict_users,
        output_path,
    )
    logger.debug(f"\nModel architecture: {server.global_model.state_dict}\n")

    # begin model training
    server.start_training()


if __name__ == "__main__":
    start_time = time.time()
    tracemalloc.start()
    psutil.cpu_percent()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f}s")
