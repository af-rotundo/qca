import pytest
import numpy as np
import random

from qca.util.util import *
from qca.chain_qw.delta_exp1 import *
from qca.util.util import x_to_index

N_RAND = 5
L_MAX = 20
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_eigenfun(execution_number):
    # we test that eigenfun from Exp1 solves the recursion relations 
    # consider a random instance of the problem 
    L = 2*random.randint(2, int(L_MAX/2))
    phi = 2 * np.pi * random.random()
    theta = 2 * np.pi * random.random()
    gamma = 2 * np.pi * random.random()
    qw = DeltaExp1(L=L, theta=theta, phi=phi, gamma=gamma)
    # first we consider the free theory (without V)
    # we build a generic eigenstate with energy omega in [0, pi]
    k = np.pi*random.random()/2
    ks = [k, -k, k-np.pi, np.pi-k]
    c = np.random.rand(4) + 1j*np.random.rand(4)
    qw.psi = qw.eigenfun(sign=1, k=k, c=c)
    alpha = np.cos(theta)
    beta = np.sin(theta)
    c = np.cos(phi)
    s = np.sin(phi)
    omega = Exp1.get_omega(sign=1, k=k, alpha=alpha)
    psi_plus = qw.psi[:2*L]
    psi_minus = qw.psi[2*L:]
    for x in range(-L+2, L):
        eq1 = (np.exp(1j*omega)*psi_plus[x_to_index(x, L)]
            -alpha*psi_plus[x_to_index(x-2, L)]
            -1j*beta*psi_minus[x_to_index(x-1, L)])
        if x == 1:
            diff = np.abs(eq1 - 1j*beta*(
                    1j*s*np.exp(1j*gamma)*psi_plus[x_to_index(0, L)]
                    +(c-1)*psi_minus[x_to_index(0, L)]))
            if diff > EPS:
                print(f'x=1, diff = {diff}')
        elif x== 2:
            diff = np.abs(eq1 - alpha*(
                    (c-1)*psi_plus[x_to_index(0, L)]
                    +1j*0*np.exp(-1j*gamma)*psi_minus[x_to_index(0, L)]))
            if diff > EPS:
                print(f'x=2, diff = {diff}')
        else:
            assert np.abs(eq1) < EPS, f"x = {x}, eq1 = {eq1}"
    for x in range(-L, L-2):
        eq2 = (np.exp(1j*omega)*psi_minus[x_to_index(x, L)]
            -alpha*psi_minus[x_to_index(x+2, L)]
            -1j*beta*psi_plus[x_to_index(x+1, L)])
        if x == -1:
            diff = np.abs(eq2 - 1j*beta*(
                    (c-1)*psi_plus[x_to_index(0, L)]
                    +1j*s*np.exp(-1j*gamma)*psi_minus[x_to_index(0, L)]))
            if diff > EPS:
                print(f'x=-1, diff = {diff}')
        elif x== -2:
            diff = np.abs(eq2 - alpha*(
                    1j*s*np.exp(1j*gamma)*psi_plus[x_to_index(0, L)]
                    +(c-1)*psi_minus[x_to_index(0, L)]))
            if diff > EPS:
                print(f'x=-2, diff = {diff}')
        else:
            assert np.abs(eq2) < EPS, f"x = {x}, eq1 = {eq2}"