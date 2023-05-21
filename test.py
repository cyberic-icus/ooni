from numpy import ndarray

from RungaKutta.AdaptiveODESolver import *
from RungaKutta.Numerical import *
from RungaKutta.Implicit import *
import numpy as np


def rhs(t, u):
    return u


def exact(t):
    return np.exp(t)


odesolvers = [ForwardEuler(rhs), ExplicitMidpoint(rhs), RungeKutta4(rhs)]
adaptiveodesolvers = [EulerHeun(rhs), RKF45(rhs), TR_BDF2_Adaptive(rhs)]
implicitodesolvers = [BackwardEuler(rhs), CrankNicolson(rhs), ImpMidpoint(rhs), Radau2(rhs), Radau3(rhs), SDIRK2(rhs), BDF_TR2(rhs)]
p = odesolvers + adaptiveodesolvers + implicitodesolvers
for i in p:
    print(type(i).__name__)

def test(solver):
    print(type(solver).__name__)
    solver.set_initial_condition(1.0)
    T = 0.1
    t_span = (0, T)
    N = 30
    print('Time step (dt) Error (e)    e/dt')
    for _ in range(10):
        t, u = solver.solve(t_span, N)
        dt = T / N
        e = abs(u[-1] - exact(T))
        e = e[0] if isinstance(e, ndarray) else e
        e = e.tolist() if isinstance(e, list) else e
        print(f'{dt:<14.7f} {e:<12.7f} {e / dt:5.4f}')
        N = N * 2
    print("\n")

for solver in odesolvers:
    test(solver)

for solver in adaptiveodesolvers:
    test(solver)

for solver in implicitodesolvers:
    test(solver)