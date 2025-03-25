from abc import ABC, abstractmethod


class Attacker(ABC):
    @abstractmethod
    def launch_attack(self, data_idx: int, **kwargs) -> dict:  # type: ignore
        """
        Launch the attack on the image specified by its index.
        Returns a dictionary containing attack results.
        """
        pass
