import numpy as np

from qca.chain_qw.simple_qw import SimpleQW
from qca.util.potential_builders import get_simple_V0, get_step_V
from qca.util.util import normalize

class DeltaSimpleQW(SimpleQW):
    def __init__(self, 
            L: int, 
            theta: float, 
            phi: float, 
            gamma: float, 
            psi: np.ndarray | None = None
        ):
        """SimpleQW with potential localized at x=0 acting on the the internal dof as 
    
            V = [[c,    i*s*e^{-i*gamma}], 
                 [i*s*e^{i*gamma},  c]]
        
        where c = cos(phi), s = sin(phi). 

    Args:
        L (int): the chain has size 2.
        theta (float): determines relative weight of diagonal and off-diagonal terms in W, through alpha = cos(theta) and beta = sin(theta). 
        phi (float): determines relative weight of diagonal and off-diagonal terms in V, through c = cos(phi) and s = sin(phi). 
        gamma (float): determines the relative phase introduced by the potential.
        psi (np.ndarray | None): state of the walker. Defaults to None. 
        """
        V0 = get_simple_V0(phi=phi, gamma=gamma)
        V = get_step_V(L=L, V0=V0, a=0, b=0)
        super().__init__(L=L, theta=theta, V=V, psi=psi)
        self.c = np.cos(phi)
        self.s = np.sin(phi)
        self.gamma = gamma

    def eigenfun(self, sign: int, k: float, c: list[np.ndarray]):
        """Builds an eigenfunction of the interacting theory given for x<0 by a superposition with the 4 degenerate states with momenta k, k-pi, -k, -k+pi. The weights for x>0 are computed from c by solving the junction conditions.
        
        The relative weights are determinded by the vector c. The band is determined by sign.

        The solution is valid on the infinite line, boundary effects on the circle spoil it. 

        Args:
            sign (int): which band to consider
            k (float): momentum
            c (list[np.ndarray]): relative weights of the 4 modes with energy equal to omega(k)

        Returns:
            _type_: unnormalized interacting eigenfunction
        """
        gp = SimpleQW.get_gp(sign, k, self.alpha)
        gm = SimpleQW.get_gp(sign, -k, self.alpha)
        S = self._generate_S(sign, k, gp, gm)
        cp = S @ c
        xL = np.arange(-self.L, 1)
        xR = np.arange(1, self.L)
        psi_plus_L = 1j*self.beta * (
            np.exp(-1j*k*xL) * (c[0] + ((-1)**np.abs(xL))*c[1]) 
            + np.exp(1j*k*xL) * (c[2] + ((-1)**np.abs(xL))*c[3])
        )
        psi_plus_R = 1j*self.beta * (
            np.exp(-1j*k*xR) * (cp[0]+(-1)**np.abs(xR)*cp[1]) 
            + np.exp(1j*k*xR) * (cp[2]+(-1)**np.abs(xR)*cp[3])
        )
        psi_plus = np.concatenate([psi_plus_L, psi_plus_R])
        xL = np.arange(-self.L, 0)
        xR = np.arange(0, self.L)
        psi_minus_L = (np.exp(-1j*k*xL) * (c[0]-c[1]*(-1)**np.abs(xL))*gp 
               + np.exp(1j*k*xL) * (c[2]-c[3]*(-1)**np.abs(xL))*gm)
        psi_minus_R = (np.exp(-1j*k*xR)*(cp[0]-(-1)**np.abs(xR)*cp[1])*gp
               + np.exp(1j*k*xR)*(cp[2]-(-1)**np.abs(xR)*cp[3])*gm)
        psi_minus = np.concatenate([psi_minus_L, psi_minus_R])
        psi = np.concatenate([psi_plus, psi_minus])
        return normalize(psi)
    
    def _generate_S(self, sign: int, k: float, gp = float, gm = float):
        """Generate matrix mapping c (weights for x<0) to cp (weights for x>0). This is a helper function for eigenfun.
        """
        # lighten notation 
        a = self.alpha
        b = self.beta
        c = self.c 
        s = self.s
        g = self.gamma
        w = self.get_omega(sign, k, self.alpha)
        MJR = np.array([
            [0, 0, 0, a*c], 
            [0, 0, -a, b*s*np.exp(-1j*g)],
            [0, np.exp(1j*w), 0, -1j*b*c],
            [np.exp(1j*w), 0, -1j*b, -1j*a*s*np.exp(-1j*g)]
            ])
        MJL = np.array([
            [np.exp(1j*w), 0, -1j*b, -1j*a*s*np.exp(1j*g)],
            [0, -np.exp(1j*w), 0, 1j*b*c],
            [0, 0, a, -b*s*np.exp(1j*g)],
            [0, 0, 0, a*c], 
            ])
        MJ = np.linalg.inv(MJR) @ MJL
        MCL = np.array([
            [np.exp(2j*k)*gp, -np.exp(2j*k)*gp, np.exp(-2j*k)*gm, -np.exp(-2j*k)*gm],
            [np.exp(1j*k)*gp, np.exp(1j*k)*gp, np.exp(-1j*k)*gm, np.exp(-1j*k)*gm],
            [1j*b*np.exp(1j*k), -1j*b*np.exp(1j*k), 1j*b*np.exp(-1j*k), -1j*b*np.exp(-1j*k)],
            [1j*b, 1j*b, 1j*b, 1j*b], 
        ])
        MCR = np.array([
            [1j*b*np.exp(-2j*k), 1j*b*np.exp(-2j*k), 1j*b*np.exp(2j*k), 1j*b*np.exp(2j*k)],
            [1j*b*np.exp(-1j*k), -1j*b*np.exp(-1j*k), 1j*b*np.exp(1j*k), -1j*b*np.exp(1j*k)],
            [np.exp(-1j*k)*gp, np.exp(-1j*k)*gp, np.exp(1j*k)*gm, np.exp(1j*k)*gm],
            [gp, -gp, gm, -gm], 
        ])
        return np.linalg.inv(MCR) @ MJ @ MCL