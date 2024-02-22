import pytest
import random 

from one_particle import *

N_RAND = 5
L_MAX = 100

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

