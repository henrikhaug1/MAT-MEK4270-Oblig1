import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    def __init__(self):
        self._dt = None
        self.c = 1.0
        self.cfl = 0.5
        self.N = None
        self.Lx = 1.0
        self.Ly = 1.0
        self.dx = None
        self.dy = None
        self.mx = None
        self.my = None

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.dx = self.Lx / self.N
        self.dy = self.Ly / self.N
        x = np.linspace(0, self.Lx, N + 1)
        y = np.linspace(0, self.Ly, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing="ij", sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        if self.mx is None or self.my is None:
            return ValueError("mx or my is None")
        self.kx = self.mx * np.pi
        self.ky = self.my * np.pi
        return self.c * np.sqrt(self.kx**2 + self.ky**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """

        self.create_mesh(N)
        self.mx, self.my = mx, my

        u_exact = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")
        U_n = u_exact(self.xij, self.yij, 0.0)

        D = self.D2(self.N)
        L_U_n = (D @ U_n) / (self.dx**2) + (U_n @ D.T) / (self.dy**2)
        U_nm1 = U_n + 0.5 * (self.c**2) * (self.dt**2) * L_U_n

        for A in (U_n, U_nm1):
            A[0, :] = A[-1, :] = 0.0
            A[:, 0] = A[:, -1] = 0.0

        self.U_n = U_n
        self.U_nm1 = U_nm1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.dx / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_func = sp.lambdify((x, y), self.ue, "numpy")
        U_exact = u_func(self.xij, self.yij)
        error = u - U_exact
        return np.sqrt(np.sum(error**2) * (self.h**2))

    def apply_bcs(self, D):
        # Zero bounadray fence around whole matrix
        D[0, :] = 0
        D[:, 0] = 0
        D[-1, :] = 0
        D[:, -1] = 0
        return D

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.c = c

        self.initialize(N, mx, my)
        U_n = self.U_n
        U_nm1 = self.U_nm1

        D = self.D2(N)
        D = self.apply_bcs(D)

        U_exact = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")

        if store_data > 0:
            out = {0: U_n.copy()}
        elif store_data == -1:
            Ue0 = U_exact(self.xij, self.yij, 0.0)
            e0 = U_n - Ue0
            errors = [np.sqrt(np.sum(e0**2) * self.dx * self.dy)]

        for n in range(1, Nt + 1):
            LUn = (D @ U_n) / (self.dx**2) + (U_n @ D.T) / (self.dy**2)
            Unp1 = 2 * U_n - U_nm1 + (self.c**2) * (self.dt**2) * LUn

            Unp1[0, :] = Unp1[-1, :] = 0.0
            Unp1[:, 0] = Unp1[:, -1] = 0.0

            U_nm1, U_n = U_n, Unp1

            if store_data > 0 and (n % store_data == 0):
                out[n] = U_n.copy()
            elif store_data == -1:
                tn = n * self.dt
                Ue = U_exact(self.xij, self.yij, tn)
                e = U_n - Ue
                errors.append(np.sqrt(np.sum(e**2) * self.dx * self.dy))

        if store_data > 0:
            return out
        elif store_data == -1:
            return self.dx, np.array(errors)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m + 1, 1)
        ]
        return r, np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N):
        raise NotImplementedError

    def ue(self, mx, my):
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError


def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    raise NotImplementedError
