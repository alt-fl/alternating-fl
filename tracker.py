from collections import deque
from copy import deepcopy, copy
from pathlib import Path
from typing import Any

import torch


class _Limited:
    """
    Represent a list but with a max capacity. If we attempt to append more
    items than its capacity, then the earliest items will be dropped
    """

    def __init__(self, capacity: int) -> None:
        self._list = deque()
        self.capacity = capacity

    def append(self, item) -> None:
        if len(self._list) >= self.capacity:
            self._list.popleft()
        self._list.append(item)

    def retrieve(self) -> deque:
        return deepcopy(self._list)


class Tracker:
    def __init__(
        self, basic: list[str], limited: list[tuple[str, int]], output_path: Path | str
    ) -> None:
        """
        A basic tracker to record resources.

        The basic types must be disjoint from the limited types, otherwise
        weird things may happen.

        There is no way to remove entry in the tracker. Well, at least not by
        using legal methods...

        Parameters:
            basic: a list of keys/types of the basic resources to be tracked
            limited: similar to basic, but also need to specify how many of the
                     same resource can be tracked
            output_path: just a path to a file
        """
        self.records: dict[str, list[Any] | _Limited] = {t: [] for t in basic}
        for t, cap in limited:
            if t in self.records:
                raise ValueError("the basic types and limited types must be disjoint")
            self.records[t] = _Limited(cap)
        self.output_path = output_path

    def track(self, type: str, item: Any, default: Any = 0) -> None:
        if type not in self.records:
            raise ValueError(f"{type!r} is not a specified type to be tracked")
        return self.records[type].append(item if item is not None else default)

    def save(self) -> None:
        """
        Use torch.save() to save the resources recorded so far
        """
        records = {}
        for t, rec in records:
            if isinstance(rec, _Limited):
                records[t] = list(rec.retrieve())
                continue
            records[t] = rec
        torch.save(records, self.output_path)
