import numpy as np

from chain_qw import ChainQW
from util import plane_wave, get_translation, get_W_from_blocks

class SimpleQW(ChainQW):
    def __init__(self, 
            L: int, 
            theta: float, 
            V: np.ndarray | None = None,
            psi: np.ndarray | None = None
        ):
        """Generate a quantum walker on a circle with internal dof of dimension 2. The walker unitary is given by 

            W = [[alpha T^2, i*beta*T, 
                 [i*betaT^{-1}, alpha T^{-2}]]

        where T is the translation operator, alpha = cos(theta), and beta = sin(theta). The potential interaction should be diagonal, but it's otherwise unspecified.

    Args:
        L (int): the chain has size 2L.
        theta (float): determines relative weight of diagonal and off-diagonal terms in W, through alpha = cos(theta) and beta = sin(theta). 
        V (np.ndarray | None, optional): diagonal potential. Defaults to None.
        psi (np.ndarray | None): state of the walker. Defaults to None. 
        """
        W = SimpleQW._get_simple_W(L=L, theta=theta)
        super().__init__(L=L, W=W, V=V, V_diagonal=False, d=2, psi=psi)
        self.alpha = np.cos(theta)
        self.beta = np.sin(theta)
    
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
