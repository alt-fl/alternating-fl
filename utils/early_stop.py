from typing import Any


def should_early_stop(idx: int, accs: Any, patience=10, delta=0.001) -> bool:
    if idx + 1 < patience:
        # train for at least some rounds according to patience
        return False

    for i in range(patience, 0, 1):
        if (accs[idx - i + 1] - accs[idx - i]) <= delta:
            return False
    return True
