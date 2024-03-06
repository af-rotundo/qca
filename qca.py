import numpy as np
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
from one_particle import plane_wave, normalize, get_translation, get_W_from_blocks

RESET = 10

class QCA():
    """Template class for quantum cellular automata (QCA).
    """
    def __init__(
        self,
        U: np.ndarray,
        psi: np.ndarray | None = None,
    ):
        self.psi = psi
        assert U.shape[0] == U.shape[1], 'U should be a square matrix'
        assert U.shape[0] > 0, 'U has size zero'
        assert np.allclose(np.eye(U.shape[0]), U.T.conj() @ U), 'U not unitary'
        assert np.allclose(np.eye(U.shape[0]), U @ U.T.conj()), 'U not unitary'
        self.U = U
        self._counter = 0
    
    def evolve(self, steps: int = 1):
        assert self.psi is not None, 'first initialize psi'
        for _ in range(steps):
            self._counter += 1
            # increase numerical stability
            if self._counter == RESET:
                self.psi = normalize(self.psi)
                self._counter = 0
            self.psi = self.U @ self.psi

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
            d (int, optional): size of the internal dof Hilbert space. Defaults to 2.
            psi (np.ndarray | None): internal state of the walker. Defaults to None.
        """
        if V is None:
            U = W
        else:
            U = W@V
        super().__init__(U=U, psi=psi)
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
    
    def plot_x(self, x_min: int | None = None, x_max: int | None = None) -> plt.plot:
        """Plot the probability of finding the walker at different positions of the chain.

        Returns:
            plt.plot: plot of the position probability
        """
        if x_min == None:
            x_min = -self.L
        if x_max == None:
            x_max = self.L
        ps = [self.p_x(x) for x in range(x_min, x_max)]
        return plt.plot(range(x_min, x_max), ps)

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

class SimpleQW(ChainQW):
    def __init__(self, 
            L: int, 
            theta: float, 
            phi: float, 
            gamma: float, 
            psi: np.ndarray | None = None
        ):
        """Generate a quantum walker on a circle with internal dof of dimension 2. The walker unitary is given by 

            W = [[alpha T^2, i*beta*T, 
                 [i*betaT^{-1}, alpha T^{-2}]]

        where T is the translation operator, alpha = cos(theta), and beta = sin(theta). The interaction is localized at x=0, it rotates the internal dof as 
    
            V = [[c,    i*s*e^{-i*gamma}], 
                 [i*s*e^{i*gamma},  c]]
        
        where c = cos(phi), s = sin(phi). 

    Args:
        L (int): the chain has size 2
        theta (float): determines relative weight of diagonal and off-diagonal terms in W, through alpha = cos(theta) and beta = sin(theta). 
        phi (float): determines relative weight of diagonal and off-diagonal terms in V, through c = cos(phi) and s = sin(phi). 
        psi (np.ndarray | None): state of the walker. Defaults to None. 
        """
        W = SimpleQW._get_simple_W(L=L, theta=theta)
        V = SimpleQW._get_delta_V(L=L, phi=phi, gamma=gamma)
        self.alpha = np.cos(theta)
        self.beta = np.sin(theta)
        self.c = np.cos(phi)
        self.s = np.sin(phi)
        self.gamma = gamma     
        super().__init__(L=L, psi=psi, W=W, V=V)
    
    def free_eigenfun(self, sign: int, k:float) -> np.ndarray:
        """Generate a unnormalized eigenfunction of the free theory, with momentum 'k' and sign specified by 'sign'. 

        The wave function is given in the basis given by the internal dof basis tensor the position basis. 

        Args:
            sign (int): specifies which band to consider 
            k (float): momentum 

        Returns:
            np.ndarray: unnormalized wave function 
        """
        if self.beta != 0:
            v_int = np.array([1j*self.beta, SimpleQW.get_gp(sign, k, self.alpha)])
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
            else:
                raise ValueError('sign should be 1 or -1.')
        return np.kron(v_int, plane_wave(self.L, k=k))

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
        omega = SimpleQW.get_omega(sign, k, alpha)
        return np.exp(1j*(omega-k))-alpha*np.exp(1j*k)
    
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

    @staticmethod
    def _get_delta_V(L:int, phi: float, gamma: float) -> np.ndarray:
        """Generate unitary for delta potential localized at x=0, which rotates the internal dof with
        
                V = [[c,    i*s*e^{-i*gamma}], 
                    [i*s*e^{i*gamma},  c]]
            
        where c = cos(phi), s = sin(phi).

        Args:
            L (int): the chain has size 2
            phi (float): sets relative weight of diagonal and off-diagonal terms in V through c = cos(phi), s = sin(phi)
            gamma: phase

        Returns:
            ndarray: potential matrix
        """
        c = np.cos(phi)
        s = np.sin(phi)
        V0 = [[c, 1j*s*np.exp(-1j*gamma)],
              [1j*s*np.exp(1j*gamma), c]]
        # first we build the potential for H_x x H_int 
        blocks = L*[np.eye(2)] + [V0] + (L-1)*[np.eye(2)]
        V = block_diag(*blocks)
        # we permute the two tensor factors
        V = V.reshape([2*L, 2, 2*L, 2])
        V = V.transpose([1, 0, 3, 2])
        V = V.reshape([4*L, 4*L])
        return V

    @staticmethod
    def _get_simple_W(L: int, theta: float) -> np.ndarray:
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

        