import numpy as np

from qca.chain_qw.chain_qw import ChainQW
from qca.util.util import get_translation, get_W_from_blocks

class Exp1(ChainQW):
    def __init__(self, 
            L: int, 
            theta: float, 
            V: np.ndarray | None = None,
            psi: np.ndarray | None = None
        ):
        """Generate a quantum walker on a circle with internal dof of dimension 2. The walker unitary is given by 

            W = [[alpha T^2, i*beta*T, 
                 [i*betaT^{-1}, alpha T^{-2}]]

        where T is the translation operator, alpha = cos(theta), and beta = sin(theta). The potential interaction should be provided by the user.

    Args:
        L (int): the chain has size 2L.
        theta (float): determines relative weight of diagonal and off-diagonal terms in W, through alpha = cos(theta) and beta = sin(theta). 
        V (np.ndarray | None, optional): diagonal potential. Defaults to None.
        psi (np.ndarray | None): state of the walker. Defaults to None. 
        """
        W = Exp1._get_W(L=L, theta=theta)
        super().__init__(L=L, W=W, V=V, V_diagonal=False, d=2, psi=psi)
        self.alpha = np.cos(theta)
        self.beta = np.sin(theta)
    
    @staticmethod
    def get_omega(sign: int, k: float, alpha: float):
        """Computes the energy corresponding to momentum 'k' in the positive or negative band, depending on the value of sign, which should be +1 or -1."""
        assert sign in [+1, -1], "sign should be +1 or -1."
        if alpha == 1:
            omega = sign*2*k
        else:
            omega = sign * np.arccos(alpha * np.cos(2*k))
        return omega

    @staticmethod
    def get_gp(sign: int, k: float, alpha: float):
        """Computes e^{i(omega_{±} - k)} - alpha e^{ik}. 
        
        Depending on the value of 'sign', either omega_+ or omega_- is computed.  
        """
        omega = Exp1.get_omega(sign, k, alpha)
        return np.exp(1j*(omega-k))-alpha*np.exp(1j*k)
    
    def _get_v_int(self, sign: int, k: float) -> np.ndarray:
        """Compute the internal part of the free theory eigenfunction. 

        Args:
            sign (int): specifies which band to consider 
            k (float): momentum 

        Returns:
            np.ndarary: internal part of the free theory eigenfunction
        """
        if self.beta != 0:
            v_int = np.array([1j*self.beta, Exp1.get_gp(sign, k, self.alpha)])
        # the case beta = 0 needs to be taken care separately 
        else:
            if sign == 1:
                if 0 <= k <= np.pi/2 or  -np.pi <= k <= -np.pi/2:
                    v_int = np.array([1, 0])
                elif -np.pi/2 < k < 0 or np.pi/2 < k <= np.pi:
                    v_int = np.array([0, 1])
                else: 
                    raise ValueError
            elif sign == -1:
                if 0 <= k <= np.pi/2 or  -np.pi <= k <= -np.pi/2:
                    v_int = np.array([0, 1])
                else: 
                    v_int = np.array([1, 0])
        return v_int

    @staticmethod
    def _get_W(L: int, theta: float) -> np.ndarray:
        """Generate a one-particle walk unitary of the form 

            W = [[alpha T^2, i*beta*T, 
                [i*betaT^{-1}, alpha T^{-2}]]

            where T is the translation operator, alpha = cos(theta), and beta = sin(theta).

        Args:
            L (int): the chain has size 2L
            theta (float): angle determining the weight of the off-diagonal terms in walk unitary

        Returns:
            np.ndarray: walk unitary
        """
        T1 = get_translation(L=L, k=1)
        T2 = get_translation(L=L, k=2)
        alpha = np.cos(theta)
        beta = np.sin(theta)
        blocks = {
            (0,0): alpha*T2, 
            (0,1): 1j*beta*T1, 
            (1,0): 1j*beta*T1.T, 
            (1,1): alpha*T2.T}
        return get_W_from_blocks(d=2, blocks=blocks)
