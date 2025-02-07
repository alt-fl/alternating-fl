from abc import ABC, abstractmethod

from math import e, log10
from typing import Type


class EpochTransition(ABC):
    @abstractmethod
    def estimate_epoch(self, round_num: int) -> int:
        pass


class NoTransition(EpochTransition):
    def __init__(self, epoch=5) -> None:
        self.epoch = epoch

    def estimate_epoch(self, round_num: int) -> int:
        return self.epoch


class BaseTransition(EpochTransition):
    def __init__(self, epoch_max: int, epoch_min=5, n=10) -> None:
        self.epoch_max = epoch_max
        self.epoch_min = epoch_min
        self.n = n


class LinearTransition(BaseTransition):
    def estimate_epoch(self, round_num: int) -> int:
        a, b, n = self.epoch_max, self.epoch_min, self.n
        if round_num >= n:
            return self.epoch_min

        est = a - (a - b) * round_num / n
        return int(est)


class QuadraticTransition(BaseTransition):
    def estimate_epoch(self, round_num: int) -> int:
        a, b, n = self.epoch_max, self.epoch_min, self.n
        if round_num >= n:
            return self.epoch_min

        est = a - (a - b) * (round_num / n) ** 2
        return int(est)


class ExponentialTransition(BaseTransition):
    def estimate_epoch(self, round_num: int) -> int:
        a, b, n = self.epoch_max, self.epoch_min, self.n
        if round_num >= n:
            return self.epoch_min

        est = (a - b) * (e**-round_num) + b
        return int(est)


class InverseVariationTransition(BaseTransition):
    def estimate_epoch(self, round_num: int) -> int:
        a, b, n = self.epoch_max, self.epoch_min, self.n
        if round_num >= n:
            return self.epoch_min

        est = b + (a - b) / (round_num + 1)
        return int(est)


class LogarithmicTransition(BaseTransition):
    def estimate_epoch(self, round_num: int) -> int:
        a, b, n = self.epoch_max, self.epoch_min, self.n
        if round_num >= n:
            return self.epoch_min

        est = a - (a - b) * log10(round_num + 1) / log10(n + 1)
        return int(est)


class EaseInTransition(EpochTransition):
    def __init__(self, epoch_max: int, epoch_min=5, n=10, c=1.0) -> None:
        self.epoch_max = epoch_max
        self.epoch_min = epoch_min
        self.n = n
        # defines the curvature of the function, when c=1, it is equivalent to
        # the linear transition
        self.c = c

    def estimate_epoch(self, round_num: int) -> int:
        a, b, n, c = self.epoch_max, self.epoch_min, self.n, self.c
        if round_num >= n:
            return self.epoch_min

        est = b + (a - b) * (1 - (round_num / n) ** c)
        return int(est)

    @staticmethod
    def estimate_max_epoch(budget: int, epoch_min: int, n: int, c: float) -> int:
        summed = sum([1 - (r / n) ** c for r in range(n)])
        est_max = (budget - epoch_min * n) / summed + epoch_min
        return int(est_max)


def get_total_epochs(trans: EpochTransition, n: int) -> int:
    return sum(map(trans.estimate_epoch, range(0, n)))


def estimate_max_epoch(
    budget: int, transition_func: Type[BaseTransition], n=10, epoch_min=5
) -> int:
    """
    Brute-force binary search for the largest max epoch without exceeding the
    specified budget for the given transition function
    """
    # initial values, the max epoch must be at least 1 and at most as much as
    # the budget
    lower = 1
    upper = budget
    last_res = -1
    while lower < upper:
        mid = (upper + lower) // 2
        trans = transition_func(mid, epoch_min=epoch_min, n=n)
        res = get_total_epochs(trans, n)
        if res == last_res:
            break

        if res <= budget:
            lower = mid
        elif res > budget:
            upper = mid - 1

        last_res = res

    return lower


def get_transition(args) -> EpochTransition:
    if not args.epoch_transition:
        # if dynamic epoch transition not enabled
        return NoTransition(args.E)

    name = args.epoch_transition
    epoch_budget, n, epoch_min = args.epoch_budget, args.transition_rounds, args.E
    curvature = args.transition_curve

    match name.lower():
        case "lin":
            trans_func = LinearTransition
        case "quad":
            trans_func = QuadraticTransition
        case "exp":
            trans_func = ExponentialTransition
        case "inv_var":
            trans_func = InverseVariationTransition
        case "log":
            trans_func = LogarithmicTransition
        case "ease_in":
            max_epoch = EaseInTransition.estimate_max_epoch(
                epoch_budget, epoch_min, n, curvature
            )
            return EaseInTransition(max_epoch, epoch_min=epoch_min, n=n, c=curvature)
        case _:
            raise ValueError(f"unknown epoch transition {name!r}")

    max_epoch = estimate_max_epoch(epoch_budget, trans_func, n=n, epoch_min=epoch_min)
    return trans_func(max_epoch, epoch_min=epoch_min, n=n)
