from matplotlib import pyplot as plt

from RungaKutta.AdaptiveODESolver import *
from RungaKutta.Numerical import *
from RungaKutta.Implicit import *


def f(t, u):
    return u

t_span = (0, 4)
N = 250

fe = EulerHeun(f)
fe.set_initial_condition(u0=1)
t1, u1 = fe.solve(t_span, N)
plt.plot(t1, u1, label="EulerHeun")

em = RKF45(f)
em.set_initial_condition(u0=1)
t2, u2 = em.solve(t_span, N)
plt.plot(t2, u2, label="RKF45")

rk4 = TR_BDF2_Adaptive(f)
rk4.set_initial_condition(u0=1)
t3, u3 = rk4.solve(t_span, N)
plt.plot(t3, u3, label="TR_BDF2_Adaptive")

rk4 = SDIRK2(f)
rk4.set_initial_condition(u0=1)
t4, u4 = rk4.solve(t_span, N)
plt.plot(t4, u4, label="SDIRK2")

time_exact = np.linspace(0, 3, 301)  # plot the exact solution in the same plot
plt.plot(time_exact, np.exp(time_exact), label='Exact')

plt.title('Адаптивные методы')
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.legend()
plt.show()
