import numpy as np
from matplotlib import pyplot as plt
from sympy.utilities.iterables import permutations
from sympy.combinatorics.permutations import Permutation
import itertools

from qca import QCA
from util import get_translation, get_W_from_blocks, plane_wave, shuffle

EPS = 1e-9

class Thirring(QCA):

    def __init__(
            self, 
            L: int,
            n_particles: int, 
            theta: float, 
            chi: float, 
            sigma_V: float | None = None,
            psi: np.ndarray | None = None
        ):
        self.n_particles = n_particles
        self.L = L
        self.theta = theta
        self.W1 = Thirring._get_one_particle_W(L, theta)
        self.V = Thirring._get_V(L, n_particles, chi, sigma_V)
        super().__init__(psi)
    
    def in_state(
        self,
        k0s: list[float], 
        x0s: list[float], 
        sigma_k_s: list[float],
        signs: list[int]
    ) ->  np.ndarray:
        """Build a initial state made out of gaussian wavepackets centered at locations specified by x0s, and momentum normally distributed around momenta specified by k0s with std deviation specified by sigma_k_s. 

        For more details on the gaussian wavepackes see the method 'one_particle_packet'. 

        The state is anti-symmetrized and normalized.

        Args:
            k0s (list[float]): mean momenta of wavepackets
            x0s (list[float]): locations of wavepackets
            sigma_k_s (list[float]): std deviation of wavepackets momenta
            signs (list[int]): specify which band the wavepackets belong to

        Returns:
            np.ndarray: array representing the state of all particles
        """
        assert len(k0s) == len(x0s) and len(k0s) == len(sigma_k_s)
        packets = [
            self.one_particle_packet(k0s[i], sigma_k_s[i], x0s[i], signs[i])
            for i in range(len(k0s))
            ]
        in_state = 1
        for packet in packets:
            in_state = np.kron(in_state, packet)
        in_state_A = Thirring._anti_symmetrize(in_state, self.n_particles,4*self.L)
        return in_state_A/np.linalg.norm(in_state_A)

    def one_particle_packet(
        self, 
        k0: float, 
        sigma_k: float, 
        x0: float, 
        sign: int
    ) -> np.ndarray:
        """Generate a one-particle wave packet center at x0, and momentum normally distributed around k0 with std deviation equal to sigma_k.

        The packet is

            |packet> = sum_k g(k) |v_k>

        where |v_k> are eigenfunctions of the free theory, and 

            g(k) = exp(-(k-k0)^2/(2*sigma_k^2)) exp(i*x0*k)

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
        return sum([
            weights[i] * self.one_particle_free_eigenfun(sign=sign, k=k) 
            for i,k in enumerate(ks)
            ])
 
    def one_particle_free_eigenfun(self, sign: int, k:float) -> np.ndarray:
        """Generate an unnormalized one-particle eigenfunction of the free theory, with momentum 'k' and sign specified by 'sign'. 

        The wave function is given in the basis given by the internal dof basis tensor the position basis. 

        Args:
            sign (int): specifies which band to consider 
            k (float): momentum 

        Returns:
            np.ndarray: unnormalized wave function 
        """
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        if s != 0:
            v_int = np.array([-1j*s, Thirring._get_gp(sign, k, c)])
        else:
            raise ValueError('not implemented yet')
        return np.kron(v_int, plane_wave(self.L, k=k))

    def get_rdm(self, i: int) -> np.ndarray:
        """Compute reduced density matrix for particle i.

        Args:
            i (int): particle index

        Returns:
            np.ndarray: reduced density matrix
        """
        n = self.n_particles
        assert 0 <= i < n, 'i should be in [0, n_particles).'
        psi_t = self.psi.reshape(n*[4*self.L])
        transposition = list(range(i))+list(range(i+1, n))+[i]
        psi_t = psi_t.transpose(transposition)
        psi_t = psi_t.reshape([(4*self.L)**(n-1), 4*self.L])
        rdm =  psi_t.T @ psi_t.conj()
        return rdm

    def get_pk(self, i: int) -> np.ndarray:
        """Returns the total probability of finding particle i with momentum k, so tracing out the internal degrees of freedom. 

        Args:
            i (int): particle index

        Returns:
            np.ndarray: probability distribution over momenta
        """
        rdm = self.get_rdm(i)
        # trace over internal dof
        rdm = rdm.reshape(2, 2*self.L, 2, 2*self.L)
        rdm = np.einsum('ijik', rdm)
        pk = []
        for n in range(-self.L, self.L):
            k = np.pi/self.L*n
            psi_k = plane_wave(self.L, k)
            pk.append(psi_k.T.conj() @ rdm @ psi_k)
        pk = np.array(pk)
        assert all(np.imag(pk) < EPS)
        return np.real(pk)
    
    def get_purity(self, i: int) -> float:
        """Compute purity of reduced density matrix for particle i.

        Args:
            i (int): particle index

        Returns:
            float: purity
        """
        rdm = self.get_rdm(i)
        return np.trace(rdm@rdm)

    def get_px(self, i):
        """Returns the total probability of finding particle i at position x, so tracing out the internal degrees of freedom. 

        Args:
            i (int): particle index

        Returns:
            np.ndarray: probability distribution over positions
        """
        rdm = self.get_rdm(i)
        # trace over internal dof
        rdm = rdm.reshape(2, 2*self.L, 2, 2*self.L)
        rdm = np.einsum('ijik', rdm)
        px = np.diag(rdm)
        assert all(np.imag(px) < EPS)
        return np.real(px)
    
    ####################################################################
    ######################### PRIVATE METHODS #########################
    ####################################################################

    def _evolve(self):
        # apply V (since this is a diagonal matrix we can use vector multiplication)
        self.psi = self.V * self.psi
        # apply W1 x W1 x ... x W1 
        self.psi = shuffle(self.psi, self.W1, self.n_particles)
    
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
    def _get_V(
        L: int, 
        n_particles: int, 
        chi: float, 
        sigma_V: float | None = None
        ) -> np.ndarray:
        """Returns potential, either contact potential if sigma_V is None (default) or smeared potential. See _get_contact_V and _get_smeared_V for more details on the potential form. 

        As in both cases the potential is a diagonal matrix, we only return the diagonal.

        Args:
            L (int): the chain has size 2L
            n_particles (int): number of particles
            chi (float): phase introduced by the potential
            sigma_V (float | None, optional): sets how smeared the potential is. Defaults to None (contact potential).

        Returns:
            np.ndarray: diagonal of potential
        """
        if sigma_V == None:
            return Thirring._get_contact_V(L, n_particles, chi)
        else:
            return Thirring._get_smeared_V(L, n_particles, chi, sigma_V)

    @staticmethod   
    def _get_contact_V(L: int, n_particles: int, chi: float) -> np.ndarray:
        """Builds a diagonal potential whose acts as 

            V |+>|x> |->|x> = e^{i*chi} |+>|x> |->|x>
            V |->|x> |+>|x> = e^{i*chi} |->|x> |+>|x>
        
        and as the identity on the other states. 

        Since the potential is diagonal, only the diagonal is returned. 

        Args:
            L (int): the chain has size 2L
            n_particles (int): number of particles
            chi (float): phase introduced by the potential

        Returns:
            np.ndarray: diagonal of potential
        """
        assert n_particles == 2, 'only implemented for n_particles = 2'
        exponents_diag = np.zeros((4*L)**n_particles)
        for i in range(2*L):
            exponents_diag[(4*L+1)*i+2*L] = 1
            exponents_diag[(4*L+1)*i+8*L**2] = 1
        diag = np.exp(1j*chi*exponents_diag)
        return diag
    
    @staticmethod
    def _get_smeared_V(
        L: int, 
        n_particles: int, 
        chi: float, 
        sigma_V: float
        ) -> np.ndarray:
        """Builds a diagonal potential whose acts as 

            V |+>|x> |->|y> = e^{i*chi(x-y)} |+>|x> |->|y>
            V |->|x> |+>|y> = e^{i*chi(x-y)} |->|x> |+>|y>

        and as the identity on the other states. Above chi(x-y) = e^{-d(x-y)^2/(2*sigma_V^2)} * chi and d(x-y) is the distance on the circle between x and y. 
        
        Since the potential is diagonal, only the diagonal is returned. 

        Args:
            L (int): the chain has size 2L
            n_particles (int): number of particles
            chi (float): phase introduced by the potential
            sigma_V (float): sets how smeared the potential is

        Returns:
            np.ndarray: diagonal of potential
        """
        assert n_particles == 2, 'only implemented for n_particles = 2'
        Vx = [
            np.exp(-Thirring._distance(L, x, y)**2/(2*sigma_V**2))
            for x,y in itertools.product(range(2*L), repeat=2)
            ]
        up = np.array([1,0])
        down = np.array([0,1])
        Vspin = np.kron(up, down) + np.kron(down, up)
        V = np.kron(Vspin, Vx)
        V = np.exp(1j*chi*V)
        V = V.reshape(2, 2, 2*L, 2*L).transpose(0, 2, 1, 3).reshape((4*L)**2)
        return V
    
    @staticmethod
    def _distance(L: int, x: int, y: int):
        """Computes distance along a circle of length 2*L."""
        d = np.abs(x-y)
        return np.min([d, 2*L-d])
    
    @staticmethod
    def _get_gp(sign: int, k: float, c: float):
        """Computes -i[sign * sin(omega(p)) + c * sin(p)]. 
        """
        omega = np.arccos(c * np.cos(k))
        return -1j*(c*np.sin(k) - sign*np.sqrt(1-c**2*np.cos(k)**2))
    
    @staticmethod
    def _anti_symmetrize(psi: np.ndarray, n: int, d: int):
        "Anti-symmetrize state psi, which should have n factors of size d. The returned state is not normalized."
        psi = psi.reshape(n * [d])
        psi_A = np.zeros(psi.shape, dtype=np.complex64)
        for perm in permutations(range(n)):
            parity = Permutation(perm).parity()
            psi_t = psi.transpose(perm)
            psi_A += (-1)**parity * psi_t
        psi_A = psi_A.reshape(d**n)
        return psi_A
