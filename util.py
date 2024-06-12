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