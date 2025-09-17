import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols("x,y")


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2) + sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        x = np.linspace(0, self.L, N + 1)
        y = np.linspace(0, self.L, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing="ij")

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags(
            diagonals=[1, -2, 1],
            offsets=[-1, 0, 1],
            shape=(self.N + 1, self.N + 1),
            format="lil",
        )
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""
        self.h = self.L / self.N
        D = self.D2() / (self.h**2)
        I = sparse.identity(self.N + 1, format="lil")

        # Kronecker sum: Laplacian = I ⊗ D + D ⊗ I
        Laplacian = sparse.kron(I, D) + sparse.kron(D, I)
        return Laplacian

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        N = self.N
        idx = np.arange((N + 1) * (N + 1)).reshape(N + 1, N + 1)

        boundary = np.unique(
            np.concatenate(
                [
                    idx[0, :],  # bottom row
                    idx[-1, :],  # top row
                    idx[:, 0],  # left column
                    idx[:, -1],  # right column
                ]
            )
        )
        return boundary

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        A = self.laplace().tolil()

        f_func = sp.lambdify((x, y), self.f, "numpy")
        b = f_func(self.xij, self.yij).astype("float").flatten()

        u_func = sp.lambdify((x, y), self.ue, "numpy")
        U_exact = u_func(self.xij, self.yij).astype(float).flatten()

        boundary = self.get_boundary_indices()

        for k in boundary:
            A.rows[k] = [k]
            A.data[k] = [1.0]
            b[k] = U_exact[k]

        return A.tocsr(), b

    def l2_error(self, u):
        """Return l2-error norm"""
        u_func = sp.lambdify((x, y), self.ue, "numpy")
        U_exact = u_func(self.xij, self.yij)
        error = u - U_exact
        return np.sqrt(np.sum(error**2) * (self.h**2))

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N + 1, N + 1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m + 1, 1)
        ]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        if x < 0 or x > self.L or y < 0 or y > self.L:
            raise ValueError("Point (x, y) is outside the domain.")

        # Find lower left index of cell (i, j) such that (x,y) is in cell
        i = int((x - 1e-14) / self.h)
        j = int((y - 1e-14) / self.h)

        # Clamp to avoid going out of bounds at upper edge
        if i >= self.N:
            i = self.N - 1
        if j >= self.N:
            j = self.N - 1

        # Local coordinates inside the cell
        xi = (x - i * self.h) / self.h
        eta = (y - j * self.h) / self.h

        # Grid values at corners
        u00 = self.U[i, j]
        u10 = self.U[i + 1, j]
        u01 = self.U[i, j + 1]
        u11 = self.U[i + 1, j + 1]

        # Bilinear interpolation formula
        return (
            (1 - xi) * (1 - eta) * u00
            + xi * (1 - eta) * u10
            + (1 - xi) * eta * u01
            + xi * eta * u11
        )


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert (
        abs(
            sol.eval(sol.h / 2, 1 - sol.h / 2)
            - ue.subs({x: sol.h / 2, y: 1 - sol.h / 2}).n()
        )
        < 1e-3
    )
