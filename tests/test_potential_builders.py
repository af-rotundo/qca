import pytest
import numpy as np
import random

from potential_builders import *

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