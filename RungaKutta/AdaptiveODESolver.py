from math import isnan, isinf

from RungaKutta.Implicit import *


class AdaptiveODESolver(ODESolver):
    def __init__(self, f, eta=0.9):
        super().__init__(f)
        self.eta = eta

    def new_step_size(self, dt, loc_error):
        eta = self.eta
        tol = self.tol
        p = self.order
        if isnan(loc_error) or isinf(loc_error):
            return self.min_dt
        new_dt = eta * (tol / loc_error) ** (1 / (p + 1)) * dt
        new_dt = max(new_dt, self.min_dt)
        return min(new_dt, self.max_dt)

    def solve(self, t_span, tol=1e-3, max_dt=np.inf, min_dt=1e-5):
        """Compute solution for t_span[0] <= t <= t_span[1]"""
        t0, T = t_span
        self.tol = tol
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.t = [t0]
        if self.neq == 1:
            self.u = [np.asarray(self.u0).reshape(1)]
        else:
            self.u = [self.u0]
        self.n = 0
        self.dt = 0.1 / np.linalg.norm(self.f(t0, self.u0))
        loc_t = t0
        while loc_t < T:
            u_new, loc_error = self.advance()
            if loc_error < tol or self.dt < self.min_dt:
                loc_t += self.dt
                self.t.append(loc_t)
                self.u.append(u_new)
                self.dt = self.new_step_size(self.dt, loc_error)
                self.dt = min(self.dt, T - loc_t, max_dt)
                self.n += 1
            else:
                self.dt = self.new_step_size(self.dt, loc_error)
        return np.array(self.t), np.array(self.u)


class EulerHeun(AdaptiveODESolver):
    def __init__(self, f, eta=0.9):
        super().__init__(f, eta)
        self.order = 1

    def advance(self):
        u, f, t = self.u, self.f, self.t
        dt = self.dt
        k1 = f(t[-1], u[-1])
        k2 = f(t[-1] + dt, u[-1] + dt * k1)
        high = dt / 2 * (k1 + k2)
        low = dt * k1
        unew = u[-1] + low
        error = np.linalg.norm(high - low)
        return unew, error


class RKF45(AdaptiveODESolver):
    def __init__(self, f, eta=0.9):
        super().__init__(f, eta)
        self.order = 4

    def advance(self):
        u, f, t = self.u, self.f, self.t
        dt = self.dt
        c2 = 1 / 4
        a21 = 1 / 4
        c3 = 3 / 8
        a31 = 3 / 32
        a32 = 9 / 32
        c4 = 12 / 13
        a41 = 1932 / 2197
        a42 = -7200 / 2197
        a43 = 7296 / 2197
        c5 = 1
        a51 = 439 / 216
        a52 = -8
        a53 = 3680 / 513
        a54 = -845 / 4104
        c6 = 1 / 2
        a61 = -8 / 27
        a62 = 2
        a63 = -3544 / 2565
        a64 = 1859 / 4104
        a65 = -11 / 40
        b1 = 25 / 216
        b2 = 0
        b3 = 1408 / 2565
        b4 = 2197 / 4104
        b5 = -1 / 5
        b6 = 0
        bh1 = 16 / 135
        bh2 = 0
        bh3 = 6656 / 12825
        bh4 = 28561 / 56430
        bh5 = -9 / 50
        bh6 = 2 / 55
        k1 = f(t[-1], u[-1])
        k2 = f(t[-1] + c2 * dt, u[-1] + dt * (a21 * k1))
        k3 = f(t[-1] + c3 * dt, u[-1] + dt * (a31 * k1 + a32 * k2))
        k4 = f(t[-1] + c4 * dt, u[-1] + dt * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t[-1] + c5 * dt, u[-1] + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t[-1] + c6 * dt, u[-1] +
               dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        low = dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5)
        high = dt * (bh1 * k1 + bh3 * k3 + bh4 * k4 + bh5 * k5 + bh6 * k6)
        unew = u[-1] + low
        error = np.linalg.norm(high - low)
        return unew, error

class AdaptiveESDIRK(AdaptiveODESolver,ESDIRK):
    def advance(self):
        b = self.b
        e = self.e
        u = self.u
        dt = self.dt
        k = self.solve_stages()
        u_step = dt * sum(b_ * k_ for b_, k_ in zip(b, k))
        error = dt * sum(e_ * k_ for e_, k_ in zip(e, k))
        u_new = u[-1] + u_step
        error_norm = np.linalg.norm(error)
        return u_new, error_norm


class TR_BDF2_Adaptive(AdaptiveESDIRK):
    def __init__(self, f, eta=0.9):
        super().__init__(f, eta)  # calls AdaptiveODESolver.__init__
        self.stages = 3
        self.order = 2
        gamma = 1 - np.sqrt(2) / 2
        beta = np.sqrt(2) / 4
        self.gamma = gamma
        self.a = np.array([[0, 0, 0],
                           [gamma, gamma, 0],
                           [beta, beta, gamma]])
        self.c = np.array([0, 2 * gamma, 1])
        self.b = np.array([beta, beta, gamma])
        bh = np.array([(1 - beta) / 3, (3 * beta + 1) / 3, gamma / 3])
        self.e = self.b - bh

