from typing import Any


def simple_moving_average(nums: Any, win_size=1, **_) -> list[float]:
    """
    Computes simple moving average (SMA) given a list of numbers.

    Parameters:
        nums: any slicable sequence of numbers, i.e. list, np.array, torch.tensor
        win_size: the window size for SMA, controls how many of the previous
                  points to take into consideration
    """
    window_val = 0
    curr_size = 0
    sma_res = []
    for i, num in enumerate(nums):
        window_val += num
        # pop the latest value from window if the window is full
        if curr_size < win_size:
            curr_size += 1
        else:
            window_val -= nums[i - win_size]
        sma_res.append(window_val / curr_size)
    return sma_res
