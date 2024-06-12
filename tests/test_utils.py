import pytest
import numpy as np
import random
from scipy.stats import unitary_group

from util import *

np.set_printoptions(suppress=True)

N_RAND = 5
L_MAX = 5
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



    