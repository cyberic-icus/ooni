from matplotlib import pyplot as plt

from RungaKutta.AdaptiveODESolver import *
from RungaKutta.Numerical import *
from RungaKutta.Implicit import *


def f(t, u):
    return u

T = 3
t_span = (0, T)
N = 10

fe = BackwardEuler(f)
fe.set_initial_condition(u0=1)
t1, u1 = fe.solve(t_span, N)
plt.plot(t1, u1, label="BackwardEuler")

em = CrankNicolson(f)
em.set_initial_condition(u0=1)
t2, u2 = em.solve(t_span, N)
plt.plot(t2, u2, label="CrankNicolson")

rk4 = ImpMidpoint(f)
rk4.set_initial_condition(u0=1)
t3, u3 = rk4.solve(t_span, N)
plt.plot(t3, u3, label="ImpMidpoint")

eh = Radau2(f)
eh.set_initial_condition(u0=1)
t4, u4 = eh.solve(t_span, N)
plt.plot(t4, u4, label="Radau2")

eh = Radau3(f)
eh.set_initial_condition(u0=1)
t4, u4 = eh.solve(t_span, N)
plt.plot(t4, u4, label="Radau3")

eh = BDF_TR2(f)
eh.set_initial_condition(u0=1)
t4, u4 = eh.solve(t_span, N)
plt.plot(t4, u4, label="BDF_TR2")

eh = SDIRK2(f)
eh.set_initial_condition(u0=1)
t4, u4 = eh.solve(t_span, N)
plt.plot(t4, u4, label="SDIRK2")

time_exact = np.linspace(0, 3, 301)  # plot the exact solution in the same plot
plt.plot(time_exact, np.exp(time_exact), label='Exact')

plt.title('Явные методы')
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.legend()
plt.show()
