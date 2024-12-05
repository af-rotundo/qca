import pytest
import numpy as np
import random

from qca.util.util import *
from qca.chain_qw.umklapp_qw import *

N_RAND = 5
L_MAX = 20
D_MAX = 3
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_get_W(execution_number):
    L = random.randint(1, L_MAX)
    T1 = get_translation(L=L, k=1)
    T2 = get_translation(L=L, k=2)
    Id = np.eye(2*L)
    W_corr = np.block(
        [[(T2-Id)/2, 1j/2*(T1+T1.T)],
         [1j/2*(T1+T1.T), (T2.T-Id)/2]
        ])
    W = UmklappQW._get_W(L)
    assert np.allclose(W, W_corr)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_free_eigenfun(execution_number):
    L = random.randint(2, int(L_MAX))
    qw = UmklappQW(L=L)
    # free solution from the infinite line stays solution on the circle only for some special values of k
    n = random.randint(-L, L-1)
    k = np.pi/L * n
    psi = normalize(qw.free_eigenfun(sign=1, k=k))
    qw.psi = psi
    steps = random.randint(1, STEPS_MAX)
    qw.evolve(steps)
    w = qw.get_omega(sign=1, k=k)
    print(f'w = {w}')
    print('distance', np.linalg.norm(qw.psi-np.exp(1j*w*steps)*psi))
    print(f'L = {L}, n = {n}, k = {k}, steps = {steps}')
    assert np.allclose(qw.psi, np.exp(1j*w*steps)*psi)   

def test_free_eigenfun_orthonorm():
    # check that the free eigenfunctions are orthonormal
    L = random.randint(2, int(L_MAX))
    qw = UmklappQW(L=L)
    dk = np.pi/L
    k1 = random.randint(-L, L-1)*dk
    sign = random.choice([1, -1])
    k2 = random.randint(-L, L-1)*dk
    psi_1 = qw.free_eigenfun(sign=sign, k=k1)
    # different momentum same sign
    psi_2 = qw.free_eigenfun(sign=sign, k=k2)
    # different sign same momentum
    psi_3 = qw.free_eigenfun(sign=-sign, k=k1)
    assert np.abs((psi_1.T.conj() @ psi_2)) < EPS
    assert np.abs((psi_1.T.conj() @ psi_3)) < EPS
