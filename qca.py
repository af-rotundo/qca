from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import block_diag
from matplotlib import pyplot as plt

from one_particle import plane_wave, normalize, get_translation, get_W_from_blocks

RESET = 10

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

