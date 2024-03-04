import pytest
import random 

from one_particle import *

N_RAND = 5
L_MAX = 100
D_MAX = 5

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



            
                    