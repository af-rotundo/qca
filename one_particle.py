# we write down a few help functions to to simulate the one-particle sector of QCA. 

import numpy as np

def get_basis_el(d: int, j: int) -> np.ndarray:
    """Generate the j-th element of the computational basis for a qudit of size d. 
    
    This is an array with j-th component set to 1 and all others set to 0.

    Args:
        d (int): size of Hilbert space
        j (int): index of basis element

    Returns:
        np.ndarray: basis vector
    """
    v = np.zeros(d)
    v[j] = 1
    return v

def get_translation(L: int, k: int = 1) -> np.ndarray:
    """Generate a numpy representation of the translation operator, acting as

            T|x> = |x+k>

        over a chain of size 2L.

    Args:
        L (int): half size of chain.
        k (int, optional): step to take in the chain. Defaults to 1.

    Returns:
        np.ndarray: translation operator
    """
    n = 2*L
    T1 = np.diag(np.ones(n-np.abs(k)), k=-np.abs(k))
    T2 = np.diag(np.ones(np.abs(k)), k=n-np.abs(k))
    T = T1 + T2
    if k < 0:
        T = T.transpose()
    return T

def get_simple_W(L: int, theta: float) -> np.ndarray:
    T1 = get_translation(L=L, k=1)
    T2 = get_translation(L=L, k=2)
    alpha = np.cos(theta)
    beta = np.sin(theta)
    return np.block(
        [[alpha*T2, 1j*beta*T1],
         [1j*beta*T1.T, alpha*T2.T]
        ]
    )

def normalize(psi: np.ndarray) -> np.ndarray:
    return psi/np.linalg.norm(psi)

def gaussian_packet(L: int, x0: int = 0, sigma: float = 1):
    x = np.arange(-L, L)
    psi = np.exp(-1/2*((x-x0)/sigma)**2)
    return normalize(psi)

def plane_wave(L: int, k: float):
    x = np.arange(-L, L)
    psi = np.exp(-1j*k*x)
    return normalize(psi)

def delta_V(L:int, chi: float):
    V = np.eye(2*(2*L), dtype=complex)
    V[3*L, 3*L] = np.exp(1j*chi)
    if L%2 == 0:
        V[2*L, 2*L] = np.exp(-1j*chi)
    else: 
        V[2*L+1, 2*L+1] = np.exp(-1j*chi)
    return V

def localized_particle(
        L: int, 
        k: float, 
        x0: int, 
        sigma: float = 1
    ) -> np.ndarray:
    x = np.arange(-L, L)
    psi = np.exp(-1j*k*x)*np.exp(-1/2*((x-x0)/sigma)**2)
    return normalize(psi)