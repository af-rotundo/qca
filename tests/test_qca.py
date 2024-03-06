import pytest
import numpy as np
import random
from scipy.stats import unitary_group

from qca import *
from one_particle import *

N_RAND = 5
L_MAX = 20
STEPS_MAX = 10
EPS = 1e-9

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_qca(execution_number):
    # test that evolve works as it's supposed to do
    L = random.randint(2, L_MAX)
    U = unitary_group.rvs(L)
    steps = random.randint(1, STEPS_MAX)
    psi = normalize(np.random.random(L) + 1j*np.random.random(L))
    qca = QCA(U=U, psi=psi)
    qca.evolve(steps=steps)
    assert np.allclose(qca.psi, np.linalg.matrix_power(U, steps)@psi)
    # check that if U is not unitary QCA raises error
    U = np.random.random([L, L])
    with pytest.raises(AssertionError) as exc_info:
        qca = QCA(U=U, psi=psi)
    exception_raised = exc_info.value
    assert type(exception_raised) == type(AssertionError())
    assert exception_raised.args == ('U not unitary',)
    # check that if U is not square QCA raises error
    U = np.random.random([L, L+1])
    with pytest.raises(AssertionError) as exc_info:
        qca = QCA(U=U, psi=psi)
    exception_raised = exc_info.value
    assert type(exception_raised) == type(AssertionError())
    assert exception_raised.args == ('U should be a square matrix',)
    # check that if U has zero size QCA raises error
    U = np.random.random([0, 0])
    with pytest.raises(AssertionError) as exc_info:
        qca = QCA(U=U, psi=psi)
    exception_raised = exc_info.value
    assert type(exception_raised) == type(AssertionError())
    assert exception_raised.args == ('U has size zero',)

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_p_x(execution_number):
    # check that position probabilities add up to 1
    L = random.randint(2, L_MAX)
    L = 2
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
    qw = SimpleQW(L=L, theta=theta, phi=0, gamma=0)
    # free solution from the infinite line stays solution on the circle only for some special values of k
    dk = np.pi/L
    # we take a random momentum between 0 and pi/2
    k = random.randint(0, int(L/2))*dk
    psi = qw.free_eigenfun(sign=1, k=k)
    qw.psi = psi
    steps = random.randint(1, STEPS_MAX)
    steps = 1
    qw.evolve(steps)
    w = qw.get_omega(sign=1, k=k, alpha=np.cos(theta))
    assert np.allclose(qw.psi, np.exp(1j*w*steps)*psi)   

@pytest.mark.parametrize("execution_number", range(N_RAND))
def test_eigenfun(execution_number):
    # we test that eigenfun from SimpleQW solves the recursion relations 
    # consider a random instance of the problem 
    L = 2*random.randint(2, int(L_MAX/2))
    phi = 2 * np.pi * random.random()
    theta = 2 * np.pi * random.random()
    gamma = 2 * np.pi * random.random()
    qw = SimpleQW(L=L, theta=theta, phi=phi, gamma=gamma)
    print('L', L)
    # first we consider the free theory (without V)
    # we build a generic eigenstate with energy omega in [0, pi]
    k = np.pi*random.random()/2
    ks = [k, -k, k-np.pi, np.pi-k]
    c = np.random.rand(4) + 1j*np.random.rand(4)
    print('k', k)
    qw.psi = qw.eigenfun(sign=1, k=k, c=c)
    alpha = np.cos(theta)
    beta = np.sin(theta)
    c = np.cos(phi)
    s = np.sin(phi)
    omega = SimpleQW.get_omega(sign=1, k=k, alpha=alpha)
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

@pytest.mark.parametrize("execution_number", range(N_RAND)) 
def test_delta_V(execution_number):
    L = random.randint(2, L_MAX)
    phi = 2*np.pi*random.random()
    c = np.cos(phi)
    s = np.sin(phi)
    gamma = 2*np.pi*random.random()
    V = SimpleQW._get_delta_V(L=L, phi=phi, gamma=gamma)
    # test that the nonzero elements of V are where they are supposed to be
    assert V[L, L] == c
    assert V[3*L, 3*L] == c
    assert V[L, 3*L] == 1j*s*np.exp(-1j*gamma)
    assert V[3*L, L] == 1j*s*np.exp(1j*gamma)