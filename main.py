from datetime import date
import os
from pathlib import Path
import re
import sys

import psutil
import time
import tracemalloc

import torch
import random
import numpy as np

from server import Server
from exp_wrapper import get_wrapper
from exp_args import ExperimentArgument


class Filter(object):
    def __init__(self, stream, re_pattern):
        self.stream = stream
        self.pattern = (
            re.compile(re_pattern) if isinstance(re_pattern, str) else re_pattern
        )
        self.triggered = False

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == "\n" and self.triggered:
            self.triggered = False
        else:
            if self.pattern.search(data) is None:
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True

    def flush(self):
        self.stream.flush()


# example
sys.stdout = Filter(
    sys.stdout, r"WARNING: The input does not fit in a single ciphertext"
)  # filter out any line which contains "Read -1" in it


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
        print(f"Client {k}: {counts}")
        summed[k] = sum(counts)
    print(f"Total samples: {sum(summed)}, {summed}")


def main():
    args = ExperimentArgument()
    wrapper = get_wrapper()
    seed_torch(args.seed)

    print(f"setting: {int(args.C * args.K)}/{args.K} active clients")

    today = str(date.today())
    filename = wrapper.get_output()
    output_dir = Path(".", today)
    output_path = Path(output_dir, filename)

    if not output_dir.exists():
        print(f"creating directory {output_dir}")
        output_dir.mkdir(parents=True)
    else:
        print(f"directory {output_dir} already exists, proceeding...")

    # split dataset into training (authentic and synthetic) and testing
    auth_data, syn_data, test_data = wrapper.get_data()
    # perform partition, and get the dictionary specifying the partition of
    # data that belongs to each  clients
    auth_dict_users, syn_dict_users = wrapper.partition_data()
    # create model
    model = wrapper.get_model()
    print("model:", args.model)

    total_params = sum(p.numel() for p in model.parameters())
    print("parameters:", total_params)

    print("==========authentic data==========")
    log_num_samples_per_class(auth_data, auth_dict_users, num_classes=args.num_classes)

    print("==========synthetic data==========")
    log_num_samples_per_class(syn_data, syn_dict_users, num_classes=args.num_classes)

    server = Server(
        model, auth_data, auth_dict_users, syn_data, syn_dict_users, output_path
    )
    print("global_model:", server.nn.state_dict)

    # begin model training
    server.fedfa_anchorloss(test_data, None, test_global_model_accuracy=True)


if __name__ == "__main__":
    start_time = time.time()
    tracemalloc.start()
    psutil.cpu_percent()
    main()
    end_time = time.time()
    print("total execution Time: ", end_time - start_time)
