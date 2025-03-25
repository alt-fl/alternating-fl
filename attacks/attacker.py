from abc import ABC, abstractmethod


class Attacker(ABC):
    @abstractmethod
    def reconstruct(self, data_idx: int | list[int], **kwargs) -> dict:  # type: ignore
        """
        Launch the attack on the image specified by its index, or a batch of
        images specified by the list of indices.
        Returns a dictionary containing attack results.
        """
        pass
