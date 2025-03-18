import logging
from datetime import date
import os
from pathlib import Path

import psutil
import time
import tracemalloc
from pympler import asizeof

import torch
import random
import numpy as np

from server import Server
from exp_wrapper import get_wrapper
from exp_args import ExperimentArgument
from he import get_he_context

from logger import logger, configure_logger, MessageContentFilter
from training.epochs import get_transition


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
    banned_phrases = [
        # this ignores tenseal warning which says some unused operations are not available
        "WARNING: The input does not fit in a single ciphertext, and some operations will be disabled.",
        "The following operations are disabled in this setup: matmul, matmul_plain, enc_matmul_plain, conv2d_im2col.",
        "If you need to use those operations, try increasing the poly_modulus parameter, to fit your input.",
        # we are not using the results in production, so secure RNG doesn't matter
        "UserWarning: Secure RNG turned off.",
        # this is something Opacus needs to fix
        "FutureWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions.",
    ]
    configure_logger(level=log_level, filters=[MessageContentFilter(banned_phrases)])

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

    logger.info(f"FL strategy: {args.strategy}")
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

    rho = args.rho_syn / args.rho_tot
    logger.info(f"Alt-FL with rho={rho:.5g} ({args.rho_syn}/{args.rho_tot})")

    client_context = None
    server_context = None
    if args.epsilon > 0:
        he_context = get_he_context()
        client_context = he_context.serialize(save_secret_key=True)
        # server should have a context, but without the secret
        # although it doesn't use this currently -.-
        server_context = he_context.serialize(save_secret_key=False)
        context_size = asizeof.asizeof(client_context)
        logger.info(f"HE enabled with epsilon={args.epsilon:.5g}")
        logger.info(f"HE context has size={context_size / 1e6:.2f}MB")

    if args.use_dp:
        logger.info(
            f"Differential Privacy enabled with target epsilon={args.dp_epsilon:.5g}"
            + f" and max gradient norm={args.max_grad_norm:.5g}"
        )

    epoch_transition = get_transition(args)
    logger.info(f"Current epoch transition strategy: {epoch_transition}")

    # initiate the server will all parameters, note that some parameters are not
    # used by the server in practice, but we pass them to server for convenience
    server = Server(
        model,
        wrapper.get_data(),
        auth_data,
        auth_dict_users,
        syn_data,
        syn_dict_users,
        output_path,
        client_context=client_context,
        server_context=server_context,
        epoch_transition=epoch_transition,
    )
    logger.debug(f"\nModel architecture: {server.global_model.state_dict}\n")

    # begin model training
    server.start_training()


if __name__ == "__main__":
    tot_time = time.time()
    tracemalloc.start()
    psutil.cpu_percent()
    main()
    tot_time = time.time() - tot_time

    minutes, seconds = divmod(tot_time, 60)
    hours, minutes = divmod(minutes, 60)
    logger.info(f"Total execution time: {'%d:%d:%d' % (hours, minutes, seconds)}")
