from abc import ABC, abstractmethod
import numpy as np

from qca import QCA
from util import x_to_index, plane_wave

RESET = 10

class ChainQW(QCA):
    """Quantum walk (QW), i.e. one-particle sector of QCA, on a 1d chain of length 2L and with internal dof of size d.

    The position x goes from -L to L-1. The walk is given by 
        U = WV
    where W is the free propagation and V is an optional interaction term.

    The order of the tensor factor is 
        H_int x H_x 
    where H_int is the internal dof Hilbert space and H_x is the position Hilbert space.
    """
    def __init__(
        self,
        L: int, 
        W: np.ndarray,
        V: np.ndarray | None = None,
        V_diagonal: bool = False,
        d: int = 2,
        psi: np.ndarray | None = None,
    ):
        """Init a QW on a line of length 2L and internal dof of size d, with  initial state psi, and optional potential V. 
        
        The free propagation W is a block matrix, whose blocks are specified by blocks_W. This should be a dictionary mapping 

            (i, j) -> W_{ij} for i,j in [d]x[d]

        If an entry is skipped, it is assumed that that block is made out of zeros. 

        Args:
            L (int): the chain has length 2L
            psi (np.ndarray): state of the QW
            W (np.ndarray): the free walk unitary
            V (np.ndarray | None, optional): potential. Defaults to None.
            V_diagonal (bool, optional): whether the potential is diagonal. Defaults to False.
            d (int, optional): size of the internal dof Hilbert space. Defaults to 2.
            psi (np.ndarray | None, optional): internal state of the walker. Defaults to None.
        """
        self.W = W
        self.V = V
        self.V_diagonal = V_diagonal
        self.L = L
        self.d = d
        super().__init__(psi=psi)

    def free_eigenfun(self, sign: int, k:float) -> np.ndarray:
        """Generate a unnormalized eigenfunction of the free theory, with momentum 'k' and sign specified by 'sign'. 

        The wave function is given in the basis given by the internal dof basis tensor the position basis. 

        Args:
            sign (int): specifies which band to consider 
            k (float): momentum 

        Returns:
            np.ndarray: unnormalized wave function 
        """
        assert sign in [-1,1], 'sign should be 1 or -1.'
        v_int = self._get_v_int(sign, k)
        return np.kron(v_int, plane_wave(self.L, k=k))
    
    @abstractmethod
    def _get_v_int(self, sign: int, k: float) -> np.ndarray:
        """Compute the internal part of the free theory eigenfunction. 

        Args:
            sign (int): specifies which band to consider 
            k (float): momentum 

        Returns:
            np.ndarary: internal part of the free theory eigenfunction
        """
        pass
    
    def wave_packet(
            self, 
            k0: float, 
            sigma_k: float, 
            x0: float, 
            sign: int
        ) -> np.ndarray:
        """Generate a wave packet center at x0, and momentum normally distributed around k0 with std deviation equal to sigma_k.

        The packet is

            |packet> = sum_k g(k) |v_k>

        where |v_k> are eigenfunctions of the free theory, and 

            g(k) = exp(-(k-k0)^2/(2*sigma_k^2)) exp(ix0k)

        The momenta are sampled from [-pi, pi) with a grid of spacing pi/L.

        Args:
            k0 (float): mean momentum of the packet
            sigma_k (float): std deviation of momentum
            x0 (float): mean position
            sign (int): label bands in the free theory

        Returns:
            np.ndarray: unnormalized wave packet
        """
        ks = np.arange(-self.L, self.L-1)*np.pi/self.L
        weights = np.exp(-(ks-k0)**2/(2*sigma_k**2)) * np.exp(1j*x0*ks)
        return sum([weights[i] * self.free_eigenfun(sign=sign, k=k) for i,k in enumerate(ks)])

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
    
    def _evolve(self):
        if self.V is not None:
            if self.V_diagonal == True:
                self.psi = self.V * self.psi
            else:
                self.psi = self.V @ self.psi
        self.psi = self.W @ self.psi
