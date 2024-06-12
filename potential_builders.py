# collection of potential builder functions

import numpy as np
from scipy.linalg import block_diag

def get_simple_V0(phi: float, gamma: float) -> np.ndarray:
    """
    Builds a matrix of the form 
    
        V0 = [[c,    i*s*e^{-i*gamma}], 
                [i*s*e^{i*gamma},  c]]
                
    where c = cos(phi), s = sin(phi). 
    
    Args:
        phi (float): sets relative weight of diagonal and off-diagonal terms in V through c = cos(phi), s = sin(phi)
        gamma: phase
        
    Returns:
        ndarray: local potential matrix
    """
    c = np.cos(phi)
    s = np.sin(phi)
    V0 = np.array([[c, 1j*s*np.exp(-1j*gamma)],
            [1j*s*np.exp(1j*gamma), c]])
    return V0 

def get_very_simple_V0(gamma: float) -> np.ndarray:
    """
    Generates a local diagonal potential of the form 
    
        V0 = [[e^{-1j*gamma}, 0],
                [0, e^{1j*gamma}]].
    
    Args:
        gamma: (half) relative phase introduced by the rotation
        
    Returns:
        ndarray: local potential matrix
    """
    V0 = np.array([[np.exp(-1j*gamma), 0],
            [0, np.exp(1j*gamma)]])
    return V0 

def get_step_V(L:int, V0: np.array, a: int, b: int) -> np.ndarray:
    """Generate unitary for a step potential, which rotates the internal dof with V0 for x in [a, b], 
    and acts as the identity otherwise. 

    Args:
        L (int): the chain has size 2L
        a (int): left end of the interval in which the potential acts. 
        b (int): right end of the interval in which the potential acts. 

    Returns:
        ndarray: potential matrix
    """
    assert a <= b, "a should be smaller than b"
    assert -L <= a <= L-1 and -L <= b <= L-1, "a and b should be in [-L, L-1]"
    # first we build the potential for H_x x H_int 
    blocks = np.abs(-L-a)*[np.eye(2)] + (1+b-a)*[V0] + (L-1-b)*[np.eye(2)]
    V = block_diag(*blocks)
    # we permute the two tensor factors
    V = V.reshape([2*L, 2, 2*L, 2])
    V = V.transpose([1, 0, 3, 2])
    V = V.reshape([4*L, 4*L])
    return V

@staticmethod
def get_smeared_V(L:int, phi: float, gamma: float, sigma_V: float) -> np.ndarray:
    """Generate unitary for a potential, which rotates the internal dof with 
        
        V0 = [[c,    i*s*e^{-i*gamma}], 
                [i*s*e^{i*gamma},  c]]
    
    for x=0, where c = cos(phi), s = sin(phi). Away from x=0 the interaction is smoothly turned of with a Gaussian envelope with standared 
    deviation equal to sigma_V. In practice we take phi(x) to be a Gaussian centered at x=0.

    Args:
        L (int): the chain has size 2L
        phi (float): sets relative weight of diagonal and off-diagonal terms in V through c = cos(phi), s = sin(phi)
        gamma: phase
        sigma_V(float): standard deviation of Gaussian envelope

    Returns:
        ndarray: potential matrix
    """
    # first we build the potential for H_x x H_int 
    blocks = [get_simple_V0(phi=phi*np.exp(-x**2/(2*sigma_V**2)), gamma=gamma) for x in np.arange(-L, L)] 
    V = block_diag(*blocks)
    # we permute the two tensor factors
    V = V.reshape([2*L, 2, 2*L, 2])
    V = V.transpose([1, 0, 3, 2])
    V = V.reshape([4*L, 4*L])
    return V   