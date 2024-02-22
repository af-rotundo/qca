import numpy as np
from matplotlib import pyplot as plt

class QCA():
    """Template class for quantum cellular automata (QCA).
    """
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
    """Quantum walk (QW), i.e. one-particle sector of QCA, on a 1d chain of length 2L and with internal dof of size d.

    The position x goes from -L to L-1. The walk is given by 
        U = WV
    where W is the free propagation and V is an optional interaction term.

    Args:
        QCA (_type_): _description_
    """
    def __init__(
        self,
        L: int, 
        psi: np.ndarray,
        blocks_w: dict[(int, int): np.ndarray],
        V: np.ndarray | None = None,
        d: int = 2
    ):
        """Init a QW on a line of length 2L and internal dof of size d, with  initial state psi, and optional potential V. 
        
        The free propagation W is a block matrix, whose blocks are specified by blocks_W. This should be a dictionary mapping 

            (i, j) -> W_{ij} for i,j in [d]x[d]

        If an entry is skipped, it is assumed that that block is made out of zeros. 

        Args:
            L (int): the chain has length 2L
            psi (np.ndarray): state of the QW
            blocks_w (dict[(int, int): np.ndarray]): non zero blocks in the free propagation
            V (np.ndarray | None, optional): potential. Defaults to None.
            d (int, optional): size of the internal dof Hilbert space. Defaults to 2.
        """
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
        """Compute the probability that the walker is found at position x. 

        Recall that position goes from x = -L to x = L-1

        Args:
            x (int): position

        Returns:
            float: probability
        """
        i_plus = x_to_index(x, self.L)
        i_minus = 2*self.L + i_plus
        return np.abs(self.psi[i_plus])**2 + np.abs(self.psi[i_minus])**2
    
    def plot_x(self) -> plt.plot:
        """Plot the probability of finding the walker at different positions of the chain.

        Returns:
            plt.plot: plot of the position probability
        """
        ps = [self.p_x(x) for x in range(-self.L, self.L)]
        return plt.plot(range(-self.L, self.L), ps)

def x_to_index(x: int, L: int) -> int:
    """Convert a position x along a chain of size 2L with x=-L, ..., L-1 to the corresponding index in the vector representation of the walker state, i=0, ..., 2L-1.

    Args:
        x (int): position
        L (int): the chain has length 2L

    Returns:
        int: index corresponding to position x
    """
    return x+L

def index_to_x(i: int, L:int) -> int:
    """Invert the map x_to_index.

    Args:
        i (int): index from 0 to 2L-1
        L (int): the chain has length 2L

    Returns:
        int: position corresponding to i
    """
    return i-L

    