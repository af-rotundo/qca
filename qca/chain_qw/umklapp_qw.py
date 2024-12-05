import numpy as np

from qca.chain_qw.chain_qw import ChainQW
from qca.util.util import get_translation, get_W_from_blocks

class UmklappQW(ChainQW):
    def __init__(self, 
            L: int, 
            V: np.ndarray | None = None,
            psi: np.ndarray | None = None
        ):
        """Generate a quantum walker on a circle with internal dof of dimension 2. The walker unitary is given by 

            W = 1/2 * [[T^2-Id, i*(T+T^{-1}), 
                      [i*(T+T^{-1}), T^{-2}-Id]]

        where T is the translation operator. The potential interaction should be provided by the user.

    Args:
        L (int): the chain has size 2L.
        V (np.ndarray | None, optional): diagonal potential. Defaults to None.
        psi (np.ndarray | None): state of the walker. Defaults to None. 
        """
        W = UmklappQW._get_W(L=L)
        super().__init__(L=L, W=W, V=V, V_diagonal=False, d=2, psi=psi)
    
    @staticmethod
    def get_omega(sign: int, k: float):
        """Computes the energy corresponding to momentum 'k' in the positive or negative band, depending on the value of sign, which should be +1 or -1."""
        assert sign in [+1, -1], "sign should be +1 or -1."
        omega = sign * np.arccos(-np.sin(k)**2)
        if k > np.pi/2 or k < -np.pi/2:  
           omega = -omega
        return omega
    
    def _get_v_int(self, sign: int, k: float) -> np.ndarray:
        """Compute the internal part of the free theory eigenfunction. 

        Args:
            sign (int): specifies which band to consider 
            k (float): momentum 

        Returns:
            np.ndarary: internal part of the free theory eigenfunction
        """
        v_int = np.array([1, -np.sin(k) + sign*np.sqrt(np.sin(k)**2+1)])
        return v_int

    @staticmethod
    def _get_W(L: int) -> np.ndarray:
        """Generate a one-particle walk unitary of the form 

             W = 1/2 * [[T^2-Id, i*(T+T^{-1}), 
                      [i*(T+T^{-1}), T^{-2}-Id]]

            where T is the translation operator, alpha = cos(theta), and beta = sin(theta).

        Args:
            L (int): the chain has size 2L

        Returns:
            np.ndarray: walk unitary
        """
        T1 = get_translation(L=L, k=1)
        T2 = get_translation(L=L, k=2)
        Id = np.eye(2*L)
        blocks = {
            (0,0): (T2-Id)/2, 
            (0,1): 1j/2*(T1+T1.T), 
            (1,0): 1j/2*(T1+T1.T), 
            (1,1): (T2.T-Id)/2}
        return get_W_from_blocks(d=2, blocks=blocks)
