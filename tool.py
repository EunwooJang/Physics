import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Root finding
def root_finding(mode, f, df, i_x_1, i_x_2, eps=1e-8):

    def bisection_method():
        x_1 = i_x_1
        x_2 = i_x_2
        array = []
        for _ in range(1000):
            x_3 = (x_1 + x_2) / 2
            if f(x_1) * f(x_3) < 0:
                x_2 = x_3
            else:
                x_1 = x_3
            array.append(x_3)
            if np.abs(x_2 - x_1) < eps:
                break
        return array

    def newton_method():
        x_1 = i_x_2
        array = []
        for _ in range(1000):
            x_2 = x_1 - f(x_1) / df(x_1)
            array.append(x_2)
            if np.abs(x_1 - x_2) < eps:
                break
            x_1 = x_2
        return array

    def secant_method():
        x_1 = i_x_1
        x_2 = i_x_2
        array = []
        for _ in range(1000):
            df = (f(x_2) - f(x_1)) / (x_2 - x_1)
            x_3 = x_2 - f(x_2) / df
            array.append(x_3)
            if np.abs(x_3 - x_2) < eps:
                break
            x_1 = x_2
            x_2 = x_3
        return array

    if mode == 1:
        return np.array(bisection_method())

    elif mode == 2:
        return np.array(newton_method())

    elif mode == 3:
        return np.array(secant_method())

    else:
        print("Invalid mode")
        return

# Curve fitting
def curve_fitting(mode, f, x, y, degree, c=1e-2, eps=1e-8, max_iter=10000, init_parm=None):

    def least_square_method():

        A = np.zeros((degree, degree))
        B = np.zeros(degree)

        for i in range(degree):
            for j in range(degree):
                A[i, j] = np.sum(x ** (2 * degree - 2 - (i + j)))

        for i in range(degree):
            B[i] = np.sum((x ** (degree - 1 - i)) * y)

        return np.linalg.solve(A, B)


    def gradient_descent_analytic():

        a = (init_parm if init_parm!=None else np.ones(degree))
        dEda_history = []
        iter = 0

        def dEda_i(a_vec, ii):
            f_x = np.sum([a_vec[k] * x ** (degree - k - 1) for k in range(degree)], axis=0)
            residual = y - f_x
            return np.sum(residual * x**(degree - ii - 1))

        delta = np.ones(degree)

        while np.sqrt(np.sum(delta**2)) > eps and iter < max_iter:
            iter += 1
            for i in range(degree):
                delta[i] = c * dEda_i(a, i)
            a -= delta
            dEda_history.append(delta.copy())

        return a, dEda_history


    def grad_descent_approx():

        a = (init_parm if init_parm!=None else np.ones(degree))

        dEda_history = []
        iter = 0

        def E(f, x, y, a_v):
            return np.sum((y - f(x, a_v)) ** 2)

        def dEda(f, x, y, a, i, da=1e-6):
            a_forward = a.copy()
            a_forward[i] += da
            a_center = a.copy()
            return (E(f, x, y, a_forward) - E(f, x, y, a_center)) / da

        delta = np.ones(degree)

        while np.sqrt(np.sum(delta ** 2)) > eps and iter < max_iter:
            iter += 1
            for i in range(degree):
                delta[i] = -c * dEda(f, x, y, a, i)
            a += delta
            dEda_history.append(delta.copy())

        return a, dEda_history


    if mode == 1:
        return least_square_method()

    elif mode == 2:
        return gradient_descent_analytic()

    elif mode == 3:
        return grad_descent_approx()

    else:
        print("Invalid mode")

# ODE SDE
class ODE:
    def __init__(self, v0, f, T, dt, x0=None):

        self.t = np.linspace(0, T, num=int(abs(T/dt)) + 1)

        self.dt = dt
        self.f = f
        self.x_enabled = x0 is not None

        self.dim = len(v0) if hasattr(v0, '__len__') else 1
        self.v = np.zeros((len(self.t), self.dim))
        self.v[0] = v0 if self.dim > 1 else [v0]

        if self.x_enabled:
            self.x = np.zeros((len(self.t), self.dim))
            self.x[0] = x0 if self.dim > 1 else [x0]
        else:
            self.x = None

    def _call_f(self, v, x, t):
        arg_count = self.f.__code__.co_argcount
        if arg_count == 1:
            return self.f(v)
        elif arg_count == 2:
            return self.f(v, x)
        elif arg_count == 3:
            return self.f(v, x, t)
        else:
            raise ValueError("f는 1~3개의 인자 (v[, x[, t]])를 가져야 합니다.")

    def Euler(self):
        for i in range(1, len(self.t)):
            t_ = self.t[i - 1]
            v_ = self.v[i - 1]
            x_ = self.x[i - 1] if self.x_enabled else None
            f_val = self._call_f(v_, x_, t_)

            self.v[i] = v_ + f_val * self.dt
            if self.x_enabled:
                self.x[i] = x_ + v_ * self.dt

        return self.t.copy(), self.v.copy(), self.x.copy() if self.x_enabled else None

    def M_Euler(self):
        for i in range(1, len(self.t)):
            t_ = self.t[i - 1]
            v_ = self.v[i - 1]
            x_ = self.x[i - 1] if self.x_enabled else None

            f1 = self._call_f(v_, x_, t_)
            vbar = v_ + f1 * self.dt
            xbar = x_ + v_ * self.dt if self.x_enabled else None
            f2 = self._call_f(vbar, xbar, t_ + self.dt)

            self.v[i] = v_ + (f1 + f2) * self.dt / 2
            if self.x_enabled:
                self.x[i] = x_ + (v_ + vbar) * self.dt / 2

        return self.t.copy(), self.v.copy(), self.x.copy() if self.x_enabled else None

    def RK2(self):
        for i in range(1, len(self.t)):
            t_ = self.t[i - 1]
            v_ = self.v[i - 1]
            x_ = self.x[i - 1] if self.x_enabled else None

            f1 = self._call_f(v_, x_, t_)
            vbar = v_ + f1 * self.dt / 2
            xbar = x_ + v_ * self.dt / 2 if self.x_enabled else None
            f2 = self._call_f(vbar, xbar, t_ + self.dt / 2)

            self.v[i] = v_ + f2 * self.dt
            if self.x_enabled:
                self.x[i] = x_ + vbar * self.dt

        return self.t.copy(), self.v.copy(), self.x.copy() if self.x_enabled else None

    def RK4(self):
        for i in range(1, len(self.t)):
            t_ = self.t[i - 1]
            dt = self.dt
            v0 = self.v[i - 1]
            x0 = self.x[i - 1] if self.x_enabled else None

            k1v = dt * self._call_f(v0, x0, t_)
            k1x = dt * v0 if self.x_enabled else None

            k2v = dt * self._call_f(v0 + k1v/2, x0 + k1x/2 if self.x_enabled else None, t_ + dt/2)
            k2x = dt * (v0 + k1v/2) if self.x_enabled else None

            k3v = dt * self._call_f(v0 + k2v/2, x0 + k2x/2 if self.x_enabled else None, t_ + dt/2)
            k3x = dt * (v0 + k2v/2) if self.x_enabled else None

            k4v = dt * self._call_f(v0 + k3v, x0 + k3x if self.x_enabled else None, t_ + dt)
            k4x = dt * (v0 + k3v) if self.x_enabled else None

            self.v[i] = v0 + (k1v + 2*k2v + 2*k3v + k4v) / 6
            if self.x_enabled:
                self.x[i] = x0 + (k1x + 2*k2x + 2*k3x + k4x) / 6

        return self.t.copy(), self.v.copy(), self.x.copy() if self.x_enabled else None

    def LeapFrog(self):
        if not self.x_enabled:
            raise ValueError("LeapFrog은 2차 방정식 (x가 있는 경우)만 지원합니다.")

        self.v[0] += 0.5 * self.dt * self._call_f(self.v[0], self.x[0], self.t[0])
        for i in range(1, len(self.t)):
            self.x[i] = self.x[i - 1] + self.v[i - 1] * self.dt
            self.v[i] = self.v[i - 1] + self._call_f(self.v[i - 1], self.x[i], self.t[i]) * self.dt

        return self.t.copy(), self.v.copy(), self.x.copy()

# PDE
# Drawing Fuction
def plot_all(X, Y, Z, x_label='X', y_label='Y', z_label='Z', Title='Plot'):

    # Contour plot
    plt.figure(figsize=(12, 6))
    plt.contour(X, Y, Z, levels=np.linspace(np.min(Z), np.max(Z), 100))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(Title + ' - Contour')
    ax = plt.gca()
    ax.set_aspect('equal')
    cb = plt.colorbar()
    cb.set_label(z_label)
    plt.show()
    plt.close()

    # Pcolor plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    cax = ax.pcolor(X, Y, Z)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(Title + ' - Pcolor')
    ax.set_aspect('equal')
    cb = plt.colorbar(cax, ax=ax)
    cb.set_label(z_label)
    plt.show()
    plt.close()

    # Surface plot
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig = plt.gcf()
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(Title + ' - Surface')
    plt.show()
    plt.close()

#FDM Method
class FDM:
    def __init__(self, domain, N):
        self.x0, self.xL = domain
        self.N = N
        self.h = (self.xL - self.x0) / (N + 1)
        self.x = np.linspace(self.x0 + self.h, self.xL - self.h, N)
        self.A = np.zeros((N, N))
        self.b = np.zeros(N)
        self.bc = []

    def add_operator(self, order, coef_func):
        if order == 2:
            self._add_second_derivative(coef_func)
        elif order == 1:
            self._add_first_derivative(coef_func)

    def _add_second_derivative(self, k_func):
        k = k_func(self.x)
        k_ext = np.zeros(self.N + 2)
        k_ext[1:-1] = k
        k_ext[0] = k[0]
        k_ext[-1] = k[-1]

        for i in range(self.N):
            k_plus = 0.5 * (k_ext[i + 2] + k_ext[i + 1])
            k_minus = 0.5 * (k_ext[i] + k_ext[i + 1])

            self.A[i, i] += -(k_plus + k_minus)
            if i > 0:
                self.A[i, i - 1] += k_minus
            if i < self.N - 1:
                self.A[i, i + 1] += k_plus

        self.A /= self.h**2

    def _add_first_derivative(self, b_func):
        b = b_func(self.x)

        for i in range(self.N):
            if i > 0:
                self.A[i, i - 1] += -b[i] / (2 * self.h)
            if i < self.N - 1:
                self.A[i, i + 1] += b[i] / (2 * self.h)

    def set_rhs(self, f_func):
        self.b = f_func(self.x)

    def set_dirichlet_bc(self, pos, val):
        if pos == self.x0:
            self.b[0] -= val / self.h**2
        elif pos == self.xL:
            self.b[-1] -= val / self.h**2
        else:
            raise ValueError("경계는 양 끝점만 설정 가능")
        self.bc.append((pos, val))

    def solve(self):
        T_interior = np.linalg.solve(self.A, self.b)
        T_full = np.zeros(self.N + 2)
        T_full[1:-1] = T_interior
        for pos, val in self.bc:
            if pos == self.x0:
                T_full[0] = val
            elif pos == self.xL:
                T_full[-1] = val
        x_full = np.linspace(self.x0, self.xL, self.N + 2)
        return x_full, T_full

    def plot(self, x, T):
        plt.plot(x, T, 'o-', label='FDM')
        plt.xlabel("x")
        plt.ylabel("T(x)")
        plt.grid(True)
        plt.show()

# FEM Method
class FEM:
    def __init__(self, domain=None, N=None, mesh=None):
        if mesh is not None:
            self.nodes = np.array(mesh)
            self.N = len(self.nodes) - 1
            self.x0 = self.nodes[0]
            self.xL = self.nodes[-1]
        elif domain is not None and N is not None:
            self.x0, self.xL = domain
            self.N = N
            self.nodes = np.linspace(self.x0, self.xL, N + 1)
        else:
            raise ValueError("Either (domain, N) or mesh must be provided")

        self.K = np.zeros((self.N+1, self.N+1))
        self.F = np.zeros(self.N+1)
        self.bc = []

    def add_operator(self, order, coef_func):
        if order == 2:
            self._add_second_derivative(coef_func)
        elif order == 1:
            self._add_first_derivative(coef_func)

    def _add_second_derivative(self, k_func):
        for i in range(self.N):
            x0, x1 = self.nodes[i], self.nodes[i+1]
            he = x1 - x0
            x_mid = (x0 + x1) / 2
            ke = k_func(x_mid)

            k_local = ke / he * np.array([[1, -1], [-1, 1]])
            self.K[i:i+2, i:i+2] += k_local

    def _add_first_derivative(self, b_func):
        for i in range(self.N):
            x0, x1 = self.nodes[i], self.nodes[i+1]
            he = x1 - x0
            x_mid = (x0 + x1) / 2
            be = b_func(x_mid)

            b_local = be / 2 * np.array([[ -1, 1], [-1, 1]])
            self.K[i:i+2, i:i+2] += b_local

    def set_rhs(self, f_func):
        for i in range(self.N):
            x0, x1 = self.nodes[i], self.nodes[i+1]
            he = x1 - x0
            x_mid = (x0 + x1) / 2
            fe = f_func(x_mid)
            f_local = fe * he / 2 * np.array([1, 1])
            self.F[i:i+2] += f_local

    def set_dirichlet_bc(self, pos, val):
        idx = 0 if pos == self.x0 else -1
        self.K[idx, :] = 0
        self.K[idx, idx] = 1
        self.F[idx] = val
        self.bc.append((pos, val))

    def solve(self):
        T = np.linalg.solve(self.K, self.F)
        return self.nodes, T

    def plot(self, x, T):
        plt.plot(x, T, 'o-', label='FEM')
        plt.xlabel("x")
        plt.ylabel("T(x)")
        plt.grid(True)
        plt.legend()
        plt.show()

# Numerical Integral
def numerical_integral(mode, f, dim, bounds, N):

    def trapezoidal_nd():
        h = [(b - a) / N for (a, b) in bounds]
        grids = [np.linspace(a, b, N + 1) for (a, b) in bounds]

        def recursive_integrate(level, point):
            if level == dim:
                weight = 1
                for i, val in enumerate(point):
                    idx = np.searchsorted(grids[i], val)
                    if idx == 0 or idx == N:
                        weight *= 0.5
                return weight * f(*point)
            else:
                total = 0
                for val in grids[level]:
                    total += recursive_integrate(level + 1, point + [val])
                return total

        integral = recursive_integrate(0, [])
        volume_element = np.prod(h)
        return integral * volume_element


    def simpson_nd():
        if N % 2 != 0:
            raise ValueError("N must be even for Simpson's Rule")

        h_list = [(b - a) / N for (a, b) in bounds]
        grids = [np.linspace(a, b, N + 1) for (a, b) in bounds]

        def recursive_loop(level, current_point):
            if level == dim:
                # 계산할 점이 완성된 경우
                weight = 1
                for i, val in enumerate(current_point):
                    idx = np.where(grids[i] == val)[0][0]
                    if idx == 0 or idx == N:
                        weight *= 1
                    elif idx % 2 == 1:
                        weight *= 4
                    else:
                        weight *= 2
                return weight * f(*current_point)
            else:
                total = 0
                for val in grids[level]:
                    total += recursive_loop(level + 1, current_point + [val])
                return total

        total_sum = recursive_loop(0, [])

        # 부피 요소
        volume_element = 1
        for h in h_list:
            volume_element *= h / 3

        return total_sum * volume_element

    if mode == 1:
        return trapezoidal_nd()
    elif mode == 2:
        return simpson_nd()
