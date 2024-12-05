from abc import ABC, abstractmethod
import numpy as np

from qca.util.util import normalize

RESET = 20

class QCA(ABC):
    """Abstract class for quantum cellular automata (QCA).
    """
    def __init__(
        self,
        psi: np.ndarray | None = None,
    ):
        self.psi = psi
        self._counter = 0
    
    def evolve(self, steps: int = 1):
        assert self.psi is not None, 'first initialize psi'
        for _ in range(steps):
            self._counter += 1
            # increase numerical stability
            if self._counter == RESET:
                self.psi = normalize(self.psi)
                self._counter = 0
            self._evolve()

    @abstractmethod
    def _evolve(self):
        pass

