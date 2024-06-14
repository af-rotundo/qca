import pytest
import numpy as np
import random
from scipy.stats import unitary_group

from util import *

np.set_printoptions(suppress=True)

N_RAND = 5
L_MAX = 5
D_MAX = 3
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_shuffle_1(execution_number):
    d = random.randint(2, 4)
    U1 = unitary_group(d).rvs()
    n = random.randint(2, 4)
    i = random.randint(0, n-1)
    dL = d**i
    dR = d**(n-i-1)
    psi_i = 1
    psi_f_correct = 1
    zero = np.array([1]+(d-1)*[0])
    for j in range(n):
        psi_i = np.kron(psi_i, zero)
        if j == i:
            psi_f_correct = np.kron(psi_f_correct, U1@zero)
        else:
            psi_f_correct = np.kron(psi_f_correct, zero)
    psi_f = shuffle_1(psi_i, dL, dR, U1)
    print(psi_f)
    print(psi_f_correct)
    assert np.allclose(psi_f, psi_f_correct)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_shuffle(execution_number):
    d = random.randint(2, 4)
    U1 = unitary_group(d).rvs()
    n = random.randint(2, 4)
    psi_i = 1
    psi_f_correct = 1
    zero = np.array([1]+(d-1)*[0])
    for i in range(n):
        psi_i = np.kron(psi_i, zero)
        psi_f_correct = np.kron(psi_f_correct, U1@zero)
    psi_f = shuffle(psi_i, U1, n)
    print(psi_f)
    print(psi_f_correct)
    assert np.allclose(psi_f, psi_f_correct)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_basis_el(execution_number):
    L = random.randint(1, L_MAX)
    j = random.randint(0, L-1)
    v = get_basis_el(L, j)
    assert v[j] == 1
    assert np.sum(np.abs(v)) == 1

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_translation(execution_number):
    L = random.randint(1, L_MAX)
    k = random.randint(-L, L)
    T = get_translation(L=L, k=k)
    # check that T is unitary
    assert np.allclose(T@T.T.conjugate(), np.eye(2*L))
    assert np.allclose(T.T.conjugate()@T, np.eye(2*L))
    j = random.randint(0, L-1)
    v0 = get_basis_el(2*L, j)
    v1 = get_basis_el(2*L, (j+k)%(2*L))
    assert np.allclose(v1, T@v0)
    T2 = get_translation(L=L, k=-k)
    assert np.allclose(T2, T.T.conjugate())

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_W_from_blocks(execution_number):
    d = random.randint(2, D_MAX)
    L = random.randint(1, L_MAX)
    blocks = {(i, i): np.eye(L) for i in range(d)}
    W = get_W_from_blocks(d, blocks)
    assert np.allclose(W, np.eye(d*L))
    X = np.zeros((L, L))
    i,j = random.randint(0, L-1), random.randint(0, L-1)
    X[i,j] = 1
    blocks = {(0,1): X}
    W = get_W_from_blocks(d, blocks)
    assert W[i, j+L] == 1
    for _ in range(10):
        k,l = random.randint(0, d-1), random.randint(0, d-1)
        if k != i or l != j+L:
            assert W[k, l] == 0




    