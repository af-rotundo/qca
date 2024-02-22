import pytest
import numpy as np
import random

from qca import *
from one_particle import get_translation, plane_wave, normalize, delta_V

N_RAND = 5

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_simple_qca(execution_number):
    # we consider a simple QW of a spin 1/2 particle with 
    # W = [[T^2, 0], [0, T^{-2}]]
    # and a (periodic) potential localized at x=0 and x=-L (x=-L+1) when L is even (odd), with form
    # V = [[1, 0], [0, exp(i*chi)]]
    L = 2*random.randint(2, 50)
    d = 2
    dk = np.pi/L
    T2 = get_translation(L=L, k=2)
    blocks_w = {(0,0): T2, (1,1): T2.T}
    c = np.random.rand(4) + 1j*np.random.rand(4)
    # first we consider the free theory (without V)
    # we build a generic eigenstate with energy omega in [0, pi]
    k = random.randint(0, L)*dk
    psi_plus = c[0]*plane_wave(L=L, k=k) + c[1]*plane_wave(L=L, k=k-np.pi)
    psi_minus = c[2]*plane_wave(L=L, k=-k) + c[3]*plane_wave(L=L, k=np.pi-k)
    psi = normalize(np.block([psi_plus, psi_minus]))
    qw = ChainQW(psi=psi, blocks_w=blocks_w, L=L, d=d)
    for _ in range(3):
        steps = random.randint(0, L)
        qw.evolve(steps=steps)
        psi *= np.exp(2j*steps*k)
        assert np.allclose(psi, qw.psi)
    # we introduce a potential 
    chi = np.pi * random.random()
    V = delta_V(L=L, chi=chi)
    # consider generalized free solutions (solutions that don't feel the potential)
    psi_plus = c[0]*plane_wave(L=L, k=k) + c[1]*plane_wave(L=L, k=k-np.pi)
    psi_minus = c[2]*plane_wave(L=L, k=-k) - c[2]*plane_wave(L=L, k=np.pi-k)
    psi = normalize(np.block([psi_plus, psi_minus]))
    qw = ChainQW(psi=psi, blocks_w=blocks_w, L=L, V=V, d=d)
    for _ in range(3):
        steps = random.randint(0, L)
        qw.evolve(steps=steps)
        psi *= np.exp(2j*steps*k)
        assert np.allclose(psi, qw.psi)
    # finally we consider "interacting" solutions
    psi_plus = c[0]*plane_wave(L=L, k=k) + c[1]*plane_wave(L=L, k=k-np.pi)
    x = np.arange(-L, L)
    psi_minus = np.exp(-1j*chi*np.heaviside(x, 1, where=x%2==0))*(c[2]*np.exp(1j*k*x)+c[3]*np.exp(1j*(k-np.pi)*x))
    psi = normalize(np.block([psi_plus, psi_minus]))
    qw = ChainQW(psi=psi, blocks_w=blocks_w, L=L, V=V, d=d)
    for _ in range(3):
        steps = random.randint(0, L)
        qw.evolve(steps=steps)
        psi *= np.exp(2j*steps*k)
        assert np.allclose(psi, qw.psi)