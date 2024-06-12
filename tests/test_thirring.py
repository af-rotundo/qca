import pytest
import numpy as np
import random
from scipy.stats import unitary_group

from thirring import *
from one_particle import *
from util import x_to_index

np.set_printoptions(suppress=True)

N_RAND = 5
L_MAX = 5
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_one_particle_W(execution_number):
    # check that W1 acts as it's supposed to on random basis elements
    L = random.randint(2, L_MAX)
    theta = np.pi*random.random()
    W1 = Thirring._get_one_particle_W(L, theta)
    assert np.allclose(W1.T.conj() @ W1, np.eye(4*L))
    spin = random.choice([0, 1])
    i = random.randint(1, 2*L-2)
    psi0 = np.kron(get_basis_el(2, spin), get_basis_el(2*L, i))
    psi = W1 @ psi0
    c = np.cos(theta)
    s = np.sin(theta)
    psi_correct = (
        c*np.kron(
            get_basis_el(2, spin), 
            get_basis_el(2*L, i+(-1)**spin)
            )+
        -1j*s*np.kron(
            get_basis_el(2, (spin+1)%2), 
            get_basis_el(2*L, i)
            )
    )
    assert np.allclose(psi, psi_correct)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_one_particle_free_eigenfun(execution_number):
    L = random.randint(2, L_MAX)
    n = 2
    theta = np.pi/4 * random.random()
    chi = 0
    qca = Thirring(L, n, theta, chi)
    # free solution from the infinite line stays solution on the circle only for some special values of k
    dk = np.pi/L
    k = random.randint(-L, L-1)*dk
    sign = random.choice([-1, -1])
    psi_i = qca.one_particle_free_eigenfun(sign, k)
    psi_i = psi_i/np.linalg.norm(psi_i)
    # check that psi_i is an eigenfunction of U with right eigenvalue
    psi_f = qca._get_one_particle_W(L, theta) @ psi_i
    omega = np.arccos(np.cos(theta) * np.cos(k))
    psi_f_correct = np.exp(sign*1j*omega) * psi_i
    assert np.allclose(psi_f, psi_f_correct)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_contact_V(execution_number):
    L = random.randint(2, L_MAX)
    chi = np.pi*random.random()
    n_particles = 2
    V = Thirring._get_contact_V(L, n_particles, chi)
    for _ in range(10*N_RAND):
        spins = [random.choice([0, 1]) for _ in range(n_particles)]
        i_s = [random.randint(1, 2*L-2) for _ in range(n_particles)]
        psi0_s = [np.kron(
            get_basis_el(2, spins[k]), get_basis_el(2*L, i_s[k])
            ) for k in range(n_particles)]
        psi0 = 1
        for k in range(n_particles):
            psi0 = np.kron(psi0, psi0_s[k]) 
        psi = V*psi0
        if spins[0] != spins[1] and i_s[0] == i_s[1]:
            assert np.allclose(psi, np.exp(1j*chi)*psi0)
        else:
            assert np.allclose(psi, psi0)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_rdm(execution_number):
    L = random.randint(2, L_MAX)
    n = 2
    theta = np.pi * random.random()
    chi = np.pi * random.random()
    qca = Thirring(L, n, theta, chi)
    ks = [np.pi * (2*random.random()-1) for _ in range(n)]
    psi_1_s = [qca.one_particle_free_eigenfun(1, k) for k in ks]
    psi_1_s = [psi_1/np.linalg.norm(psi_1) for psi_1 in psi_1_s]
    psi = 1
    for psi_1 in psi_1_s:
        psi = np.kron(psi, psi_1)
    qca.psi = psi
    i = random.choice(range(n))
    rdm = qca.get_rdm(i)
    eigs = np.linalg.eigvals(rdm)
    assert all([np.real(eig) > -EPS for eig in eigs])
    assert all([np.abs(np.imag(eig)) < EPS for eig in eigs])
    assert np.abs(np.real(sum(eigs))-1) < EPS
    rdm_correct = np.outer(psi_1_s[i], psi_1_s[i].conj())
    assert np.allclose(rdm, rdm_correct)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_anti_symmetrize(execution_number):
    n = random.randint(2, L_MAX)
    d = random.randint(2, L_MAX)
    psi_s = [unitary_group(d).rvs()[0] for _ in range(n)]
    psi = 1
    for psi_i in psi_s:
        psi = np.kron(psi, psi_i)
    psi_A = Thirring._anti_symmetrize(psi, n, d)
    psi_A = psi_A.reshape(n*[d])
    for _ in range(100):
        i = [random.randint(0, d-1) for _ in range(n)]
        p = Permutation.random(n)
        j = p(i)
        assert np.abs(psi_A[tuple(j)] - (-1)**p.parity() * psi_A[tuple(i)]) < 1e-6

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_pk(execution_number):
    # check that fourier of free eigenfunction with momentum k is correct
    L = random.randint(2, L_MAX)
    n = 2
    theta = np.pi * random.random()
    chi = np.pi * random.random()
    qca = Thirring(L, n, theta, chi)
    ms = [random.randint(0, 2*L-1) for _ in range(n)]
    ks = [np.pi/L * (m-L) for m in ms]
    states = [
        qca.one_particle_free_eigenfun(random.choice([-1,1]), k) 
        for k in ks
        ]
    psi = 1
    for state in states:
        psi = np.kron(psi, state)
    qca.psi = normalize(psi)
    i = random.choice(range(n))
    print(i)
    pk = qca.get_pk(i)
    assert np.isclose(pk[ms[i]], 1)
    pk[ms[i]] = 0
    assert np.allclose(pk, 0)

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_get_smeared_V(execution_number):
    # builds a random state and check that the potential acts as expected
    L = random.randint(2, L_MAX)
    n = 2
    chi = np.pi * random.random()
    sigma_V = L * random.random()
    V = Thirring._get_smeared_V(L, n, chi, sigma_V)
    up = np.array([1,0])
    down = np.array([0,1])
    x = random.randint(-L, L-1)
    y = random.randint(-L, L-1)
    ix = x_to_index(x, L)
    iy = x_to_index(y, L)
    ket_x = np.zeros(2*L)
    ket_x[ix] = 1
    ket_y = np.zeros(2*L)
    ket_y[iy] = 1
    psi_1 = np.kron(up, ket_x)
    psi_2 = np.kron(down, ket_y)
    if random.choice([0,1]) == 0:
        psi_in = np.kron(psi_1, psi_2)
    else:
        psi_in = np.kron(psi_2, psi_1)
    psi_out = V * psi_in
    d = Thirring._distance(L, x, y)
    phase = np.exp(1j * chi * np.exp(-d**2/(2*sigma_V**2)))
    assert np.allclose(psi_out, phase*psi_in)
