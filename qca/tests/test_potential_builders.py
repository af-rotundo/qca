import pytest
import numpy as np
import random
from scipy.stats import unitary_group

from qca.util.potential_builders import *
from qca.util.util import *

N_RAND = 5
L_MAX = 20
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_delta_V(execution_number):
    L = random.randint(2, L_MAX)
    phi = 2*np.pi*random.random()
    c = np.cos(phi)
    s = np.sin(phi)
    gamma = 2*np.pi*random.random()
    V0 = get_simple_V0(phi, gamma)
    V = get_step_V(L=L, V0=V0, a=0, b=0)
    # test that the nonzero elements of V are where they are supposed to be
    assert V[L, L] == c
    assert V[3*L, 3*L] == c
    assert V[L, 3*L] == 1j*s*np.exp(-1j*gamma)
    assert V[3*L, L] == 1j*s*np.exp(1j*gamma)

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_step_V(execution_number):
    L = random.randint(2, L_MAX)
    V0 = unitary_group.rvs(2)
    a = random.randint(-L, L-1)
    b = random.randint(a, L-1)
    V = get_step_V(L, V0, a, b)
    x = random.randint(-L, L-1)
    i = x_to_index(L, x)
    psi_x = get_basis_el(2*L, i)
    if random.random() > 1/2:
        psi_int = np.array([1,0])
    else:
        psi_int = np.array([0,1])
    psi = np.kron(psi_int, psi_x)
    if a <= x <= b:
        psi_correct = np.kron(V0@psi_int, psi_x)
    else:
        psi_correct = np.kron(psi_int, psi_x)
    assert np.allclose(psi_correct, V@psi)


@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_smeared_V(execution_number):
    L = random.randint(2, L_MAX)
    phi = np.pi * random.random()
    gamma = np.pi * random.random()
    sigma_V = L/2*random.random()
    get_V = lambda phi: get_simple_V0(phi, gamma)
    V = get_smeared_V(L, get_V, phi, sigma_V)
    x = random.randint(-L, L-1)
    i = x_to_index(L, x)
    psi_x = get_basis_el(2*L, i)
    if random.random() > 1/2:
        psi_int = np.array([1,0])
    else:
        psi_int = np.array([0,1])
    psi = np.kron(psi_int, psi_x)
    V0 = get_simple_V0(phi=phi*np.exp(-x**2/(2*sigma_V**2)), gamma=gamma)
    psi_correct = np.kron(V0@psi_int, psi_x)
    assert np.allclose(psi_correct, V@psi)