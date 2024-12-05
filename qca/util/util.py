import numpy as np
from matplotlib import pyplot as plt

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

def shuffle(psi: np.ndarray, U1: np.ndarray, n: int):
    """Apply the unitary found by tensoring U1 n times by itself, i.e.
        
        U = U1 x U1 x ... x U1 (n times)
    
    to psi. We do this using the shuffle algorithm, see e.g. http://yoksis.bilkent.edu.tr/pdf/files/11666.pdf.

    If U has dimension dxd, psi should have dimension d**n. 
    """
    d = len(U1)
    assert len(psi) == d**n, "psi and U1 don't have compatible shapes"
    # we act with Id_L x U_i x Id_R where U_i is acting on the i-th factor
    for i in range(n):
        # dimension 
        dL = d**i
        dR = d**(n-i-1)
        psi = shuffle_1(psi, dL, dR, U1)
    return psi

def shuffle_1(psi: np.ndarray, dL: int, dR: int, U1: np.ndarray) -> np.ndarray:
    """Apply Id_L x U1 x Id_R to psi, where Id_L and Id_R are identities with dimensions dL and dR, respectively.

    Args:
        psi (np.ndarray): _description_
        dL (int): _description_
        dR (int): _description_
        U1 (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    d = len(U1)
    psi = psi.reshape((dL, d, dR))
    psi = np.transpose(psi, (1,0,2))
    psi = psi.reshape(d, dL*dR)
    psi = np.matmul(U1, psi)
    psi = psi.reshape(d, dL, dR)
    psi = np.transpose(psi, (1,0,2))
    return psi.reshape(dL*d*dR)

def plot_x(qw, x_min: int | None = None, x_max: int | None = None) -> plt.plot:
    """Plot the probability of finding the walker at different positions of the chain.

    Returns:
        plt.plot: plot of the position probability
    """
    if x_min == None:
        x_min = -qw.L
    if x_max == None:
        x_max = qw.L
    ps = [qw.p_x(x) for x in range(x_min, x_max)]
    p = plt.plot(range(x_min, x_max), ps)
    plt.title('Position probability distribution')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    return p

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

def shift(v, n):
    n = n % len(v)
    if n == 0:
        return v
    vp = np.empty_like(v)
    vp[:n] = v[-n:]
    vp[n:] = v[:-n]
    return vp