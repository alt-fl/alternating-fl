from abc import ABC, abstractmethod

import math


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
    def __init__(self, epoch_min, epoch_max, dropoff_rate=1.0) -> None:
        self.epoch_min = epoch_min
        self.epoch_max = epoch_max
        self.rate = dropoff_rate


class InverseLinearTransition(BaseTransition):
    def estimate_epoch(self, round_num: int) -> int:
        a, b, r = self.epoch_max, self.epoch_min, self.rate
        est = b + (a - b) / (r * round_num)
        return int(est)


class ExponentialTransition(BaseTransition):
    def estimate_epoch(self, round_num: int) -> int:
        a, b, r = self.epoch_max, self.epoch_min, self.rate
        e = math.e
        est = b + (a - b) / e ** (r * (round_num - 1))
        return int(est)


class LinearTransition(EpochTransition):
    def __init__(self, epoch_min, epoch_max, n=10) -> None:
        self.epoch_min = epoch_min
        self.epoch_max = epoch_max
        self.n = n

    def estimate_epoch(self, round_num: int) -> int:
        a, b, n = self.epoch_max, self.epoch_min, self.n
        if round_num > n:
            return self.epoch_min
        est = a - (a - b) / n * (round_num - 1)
        return int(est)


class LogarithmicTransition(EpochTransition):
    def __init__(self, epoch_min, epoch_max, n=10) -> None:
        self.epoch_min = epoch_min
        self.epoch_max = epoch_max
        self.n = n

    def estimate_epoch(self, round_num: int) -> int:
        a, b, n = self.epoch_max, self.epoch_min, self.n
        if round_num >= n:
            return self.epoch_min

        est = -math.log10(round_num / n) * (a - b) + b
        return int(est)


def get_transition(args) -> EpochTransition:
    if not args.epoch_transition:
        # if dynamic epoch transition not enabled
        return NoTransition(args.E)

    name = args.epoch_transition
    epoch_max, epoch_min, rate = args.epoch_max, args.epoch_min, args.transition_rate

    match name.lower():
        case "inv_lin":
            return InverseLinearTransition(epoch_min, epoch_max, dropoff_rate=rate)
        case "exp":
            return ExponentialTransition(epoch_min, epoch_max, dropoff_rate=rate)
        case "lin":
            return LinearTransition(epoch_min, epoch_max, n=rate)
        case "log":
            return LogarithmicTransition(epoch_min, epoch_max, n=rate)
        case _:
            raise ValueError(f"unknown epoch transition {name!r}")
