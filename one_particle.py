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

def get_W_from_blocks(d:int, blocks: dict[(int, int): np.ndarray]):
    """Init a walker unitary, W, for a walker with internal dof of size d. W is a block matrix, whose blocks are specified by blocks_W. This should be a dictionary mapping 

            (i, j) -> W_{ij} for i,j in [d]x[d]

        If an entry is skipped, it is assumed that that block is made out of zeros.
    """
    assert all(np.array(list(blocks)).flatten() < d)
    shape = next(iter(blocks.values())).shape
    assert all([block.shape == shape for block in blocks.values()])
    zeros = np.zeros(shape)
    W = np.block(
            [[blocks[i,j] if (i, j) in blocks else zeros for j in range(d)] 
             for i in range(d)]
        )
    return W

def normalize(psi: np.ndarray) -> np.ndarray:
    """Normalize vector (using Frobenius norm).

    Args:
        psi (np.ndarray): state to normalize

    Returns:
        np.ndarray: normalized state
    """
    norm = np.linalg.norm(psi)
    if norm != 0:
        return psi/np.linalg.norm(psi)
    else:
        raise ValueError('Vector has zero norm')

def gaussian_packet(L: int, x0: int = 0, sigma: float = 1):
    x = np.arange(-L, L)
    psi = np.exp(-1/2*((x-x0)/sigma)**2)
    return normalize(psi)

def plane_wave(L: int, k: float) -> np.ndarray:
    """Prepare a plane wave state with momentum k over a chain of size 2L.

    In equation, the state (up to normalization) is 

        psi(x) = e^{-ikx}
    
    Periodic boundary conditions are preserved if k = pi/L * n for some integer n. 

    Args:
        L (int): the chain has size 2L
        k (float): momentum

    Returns:
        np.ndarray: normalized plane wave
    """
    x = np.arange(-L, L)
    psi = np.exp(-1j*k*x)
    return normalize(psi)

def localized_particle(
        L: int, 
        k: float, 
        x0: int, 
        sigma: float = 1
    ) -> np.ndarray:
    x = np.arange(-L, L)
    psi = np.exp(-1j*k*x)*np.exp(-1/2*((x-x0)/sigma)**2)
    return normalize(psi)

def shift(v, n):
    n = n % len(v)
    if n == 0:
        return v
    vp = np.empty_like(v)
    vp[:n] = v[-n:]
    vp[n:] = v[:-n]
    return vp