# We implement a small class that can simulate a QCA on a 1d chain
# For now we consider only 1d chains of size 2L+1 with periodic boundary conditions. The position labels go from x = -L to L.
import numpy as np
from matplotlib import pyplot as plt

class QCA():
    def __init__(
        self,
        psi: np.ndarray,
        U: np.ndarray,
    ):
        self.psi = psi
        self.U = U
    
    def evolve(self, steps: int = 1):
        for _ in range(steps):
            self.psi = self.U @ self.psi

class ChainQW(QCA):
    def __init__(
        self,
        L: int, 
        psi: np.ndarray,
        blocks_w: dict[(int, int): np.ndarray],
        V: np.ndarray | None = None,
        d: int = 2
    ):
        assert all(np.array(list(blocks_w)).flatten() < d)
        zeros = np.zeros([2*L, 2*L])
        W = np.block(
            [[blocks_w[i,j] if (i, j) in blocks_w else zeros for i in range(d)] 
             for j in range(d)]
        )
        if V is None:
            U = W
        else:
            U = W@V
        super().__init__(psi, U)
        self.L = L
        self.d = d

    def p_x(self, x: int) -> float:
        i_plus = x_to_index(x, self.L)
        i_minus = 2*self.L + i_plus
        return np.abs(self.psi[i_plus])**2 + np.abs(self.psi[i_minus])**2
    
    def plot_x(self) -> plt.plot:
        ps = [self.p_x(x) for x in range(-self.L, self.L)]
        return plt.plot(range(-self.L, self.L), ps)

def x_to_index(x: int, L: int) -> int:
    return x+L

def index_to_x(i: int, L:int) -> int:
    return i-L

    