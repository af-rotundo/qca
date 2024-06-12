import pytest
import numpy as np
import random
from scipy.stats import unitary_group

from one_particle import *
from simple_qw import *
from util import x_to_index

N_RAND = 5
L_MAX = 20
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_p_x(execution_number):
    # check that position probabilities add up to 1
    L = random.randint(2, L_MAX)
    W = unitary_group.rvs(4*L)
    psi = normalize(np.random.random(4*L) + 1j*np.random.random(4*L))
    qw = ChainQW(L=L, W=W, psi=psi)
    ps = [qw.p_x(x) for x in np.arange(-L, L)]
    assert abs(sum(ps)-1) < EPS
    # check that this doesn't change after evolution
    steps = random.randint(1, STEPS_MAX)
    qw.evolve(steps=steps)
    ps = [qw.p_x(x) for x in np.arange(-L, L)]
    assert abs(sum(ps)-1) < EPS
    # check probability is calculated correctly
    qw = ChainQW(L=L, W=W, psi=psi)
    x = random.randint(-L, L-1)
    j = x_to_index(x, L)
    vx = get_basis_el(2*L, j)
    proj = np.kron(np.eye(2), np.outer(vx, vx))
    px = psi.T.conj() @ proj @ psi
    assert abs(px - qw.p_x(x)) < EPS
