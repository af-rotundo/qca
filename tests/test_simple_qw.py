import pytest
import numpy as np
import random

from util import *
from simple_qw import *

N_RAND = 5
L_MAX = 20
D_MAX = 3
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_get_simple_W(execution_number):
    L = random.randint(1, L_MAX)
    theta = 2*np.pi*random.random()
    T1 = get_translation(L=L, k=1)
    T2 = get_translation(L=L, k=2)
    alpha = np.cos(theta)
    beta = np.sin(theta)
    W_corr = np.block(
        [[alpha*T2, 1j*beta*T1],
         [1j*beta*T1.T, alpha*T2.T]
        ])
    W = SimpleQW._get_simple_W(L, theta)
    assert np.allclose(W, W_corr)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_gp(execution_number):
    k = np.pi/2*random.random()
    alpha = random.random() 
    sign = random.choice([1, -1])
    gp1 = SimpleQW.get_gp(sign, k, alpha)
    gp2 = SimpleQW.get_gp(sign, k+sign*np.pi, alpha)
    assert  np.allclose(gp1, -gp2)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_shift(execution_number):
    L = random.randint(1, L_MAX)
    print('L', L)
    v = np.random.rand(L)
    print(v)
    s = random.randint(-STEPS_MAX,STEPS_MAX)
    vp = shift(v, s)
    vc = np.array([v[(i-s)%L] for i in range(L)])
    assert np.allclose(vp, vc)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_free_eigenfun(execution_number):
    L = random.randint(2, int(L_MAX))
    theta = 2 * np.pi * random.random()
    qw = SimpleQW(L=L, theta=theta)
    # free solution from the infinite line stays solution on the circle only for some special values of k
    n = random.randint(-L, L-1)
    k = np.pi/L * n
    psi = normalize(qw.free_eigenfun(sign=1, k=k))
    qw.psi = psi
    steps = random.randint(1, STEPS_MAX)
    qw.evolve(steps)
    w = qw.get_omega(sign=1, k=k, alpha=qw.alpha)
    print('distance', np.linalg.norm(qw.psi-np.exp(1j*w*steps)*psi))
    print(f'L = {L}, theta = {theta}, n = {n}, steps = {steps}')
    assert np.allclose(qw.psi, np.exp(1j*w*steps)*psi)   

def test_free_eigenfun_orthonorm():
    # check that the free eigenfunctions are orthonormal
    L = random.randint(2, int(L_MAX))
    theta = 2 * np.pi * random.random()
    qw = SimpleQW(L=L, theta=theta)
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
