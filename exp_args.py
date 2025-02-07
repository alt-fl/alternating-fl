from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any
import torch


class _Arguments:
    _args: Any = None


def ExperimentArgument():
    """
    Simple singleton for the experiment arguments, should avoid using this
    everywhere, and ideally pass around the specific arguments to the functions
    that actually need them
    """
    if _Arguments._args is None:
        _Arguments._args = _parse_args()

    return _Arguments._args


def _parse_args():
    parser = ArgumentParser()

    # basic training arguments
    parser.add_argument(
        "--E", type=int, default=5, help="(minimum) number of rounds of local training"
    )
    parser.add_argument(
        "--r", type=int, default=200, help="number of communication rounds"
    )
    parser.add_argument("--K", type=int, default=100, help="number of total clients")
    parser.add_argument("--B", type=int, default=64, help="local batch size")
    parser.add_argument("--TB", type=int, default=1000, help="test batch size")
    parser.add_argument("--C", type=float, default=0.1, help="client samspling rate")

    # experiment settings
    parser.add_argument("--output", type=str, help="name of the file to output results")
    parser.add_argument(
        "--save_every", type=int, default=20, help="save the checkpoint every n rounds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LeNet5",
        help="only LeNet5 architecture supported",
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument(
        "--dataset_path", type=str, default="data/CIFAR10/", help="path to the dataset"
    )
    parser.add_argument("--epsilon", type=float, default=0, help="selective HE ratio")
    parser.add_argument(
        "--rho_syn", type=int, default=0, help="number of synthetic rounds"
    )
    parser.add_argument(
        "--rho_tot",
        type=int,
        default=0,
        help="number of total rounds (for interleaving ratio)",
    )

    parser.add_argument(
        "--syn_balance",
        type=str,
        default="self",
        help="'self' or 'all' to balance synthetic data per se or as a whole",
    )
    parser.add_argument(
        "--auth_balance",
        action=BooleanOptionalAction,
        help="whether or not to balance authentic dataset",
    )
    parser.add_argument(
        "--init_syn_rounds",
        type=int,
        default=0,
        help="use only synthetic data for first n rounds",
    )

    # dynamic epoch transitions
    parser.add_argument(
        "--epoch_transition",
        type=str,
        choices=("lin", "quad", "exp", "inv_var", "log", "ease_in"),
        help="enable dynamic epoch transition",
    )
    parser.add_argument(
        "--epoch_budget",
        type=int,
        default=1000,
        help="budget for the dynamic epoch transitions",
    )
    parser.add_argument(
        "--transition_rounds",
        type=int,
        default=10,
        help="over how many rounds should the transition occur",
    )
    parser.add_argument(
        "--transition_curve",
        type=float,
        default=1,
        help="transition parameter for certain transition function",
    )

    # optimizer arguments
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.99,
        help="learning rate decay per global round",
    )
    parser.add_argument(
        "--lambda_anchor",
        type=float,
        default=0.1,
        help="anchor proximal term constant",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="type of optimizer (sgd or adam)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0, help="SGD momentum (default: 0)"
    )
    parser.add_argument(
        "--momentum_anchor",
        type=float,
        default=0.5,
        help="dynamic momentum update for feature anchor(default: 0.5)",
    )

    # model and data split arguments
    parser.add_argument(
        "--dims_feature", type=int, default=84, help="feature dimension"
    )  # 192
    parser.add_argument(
        "--trainset_sample_rate", type=int, default=1, help="trainset sample rate"
    )
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument(
        "--num_perclass",
        type=int,
        default=10,
        help="number of per class in one client dataset",
    )
    parser.add_argument(
        "--skew", type=str, default="label", help="distribution skew setting"
    )

    # other arguments
    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--verbose", action="store_true", help="verbose print")
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed (default: 1234)"
    )

    args = parser.parse_args()
    return args
