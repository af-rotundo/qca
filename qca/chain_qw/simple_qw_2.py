import numpy as np

from qca.chain_qw import ChainQW
from qca.util.util import get_translation, get_W_from_blocks

class SimpleQW2(ChainQW):
    def __init__(self, 
            L: int, 
            theta: float,
            nw: float,
            V: np.ndarray | None = None,
            psi: np.ndarray | None = None
        ):
        """Generate a quantum walker on a circle with internal dof of dimension 2. The walker unitary is given by 

            W = [[c^2*T^{-2} - s^2, i*s*c*(T+T^{-1}), 
                 [i*(T+T^{-1}), c^2*T^2 - s^2]]

        where T is the translation operator. The potential interaction should be provided by the user.

    Args:
        L (int): the chain has size 2L
        theta (float): parameter of evolution
        V (np.ndarray | None, optional): diagonal potential. Defaults to None.
        psi (np.ndarray | None): state of the walker. Defaults to None. 
        """
        W = SimpleQW2._get_one_particle_W(L=L, theta=theta)
        super().__init__(L=L, W=W, V=V, V_diagonal=False, d=2, psi=psi)
        self.theta = theta
        self.nw = nw

    @staticmethod
    def _get_gp(sign: int, k: float, c: float, s: float):
        """Computes -i[sign * sin(omega(p)) + c * sin(p)]. 
        """
        t = s/c
        return 1/t * (np.sin(k) + sign * np.sqrt(np.sin(k)**2 + t**2))
    
    def _get_v_int(self, sign: int, k: float) -> np.ndarray:
        """Compute the internal part of the free theory eigenfunction. 

        Args:
            sign (int): specifies which band to consider 
            k (float): momentum 

        Returns:
            np.ndarary: internal part of the free theory eigenfunction
        """
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        if s != 0:
            v_int = np.array([1, SimpleQW2._get_gp(sign, k, c, s)])
        else:
            raise ValueError('not implemented yet')
        return v_int
    
    def _evolve(self):
        if self.V is not None:
            if self.V_diagonal == True:
                self.psi = self.V * self.psi
            else:
                self.psi = self.V @ self.psi
        for i in range(self.nw):
            self.psi = self.W @ self.psi

    @staticmethod
    def _get_one_particle_W(L: int, theta: float) -> np.ndarray:
        """Builds the free evolution operator for one particle

            W1 = [[c T^*, -is],
                  [-is, c T]]
                
            where c = cos(theta), s = sin(theta), and T is the shift operator that acts as T|k> = |k+1>. 

        Args:
            L (int): the chain has size 2L
            theta (float): angle determining c = cos(theta) and s = sin(theta)

        Returns:
            np.ndarray: array corresponding
        """
        T = get_translation(L=L, k=1)
        c = np.cos(theta)
        s = np.sin(theta)
        blocks = {
            (0,0): c*T, 
            (0,1): -1j*s*np.eye(2*L), 
            (1,0): -1j*s*np.eye(2*L), 
            (1,1): c*T.T}
        return get_W_from_blocks(d=2, blocks=blocks)

    @staticmethod
    def _get_W(L: int, theta: float) -> np.ndarray:
        """Generate a one-particle walk unitary of the form 

           W = [[c^2 * T^{-2} - s^2, i*s*c*(T+T^{-1}), 
                [i*(T+T^{-1}), c^2 * T^2 - s^2]]

            where T is the translation operator, and c = cos(theta) and 
            s = sin(theta).

        Args:
            theta (float): parameter of evolution
            L (int): the chain has size 2L
        
        Returns:
            np.ndarray: walk unitary
        """
        c = np.cos(theta)
        s = np.sin(theta)
        # need to change convention for definition of translation
        T1 = get_translation(L=L, k=-1)
        T2 = get_translation(L=L, k=-2)
        Id = np.eye(2*L)
        blocks = {
            (0,0): c**2 * T2.T - s**2*Id, 
            (0,1): -1j * s * c * (T1+T1.T), 
            (1,0): -1j * s * c * (T1+T1.T), 
            (1,1): c**2 * T2 - s**2 * Id
        }
        return get_W_from_blocks(d=2, blocks=blocks)
