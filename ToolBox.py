import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

class Vector_Field:
    """Vector fields used in the project"""

    def __init__(self, k, F):
        """Hyperparameters:

        - k: Kill rate of virus
        - F: Food rate of virus and food creation
        """

        self.k = k
        self.F = F
        return None

    def f_1(self, u, v):
        """Dynamics of the food.

        Inputs:
        - u: Food
        - v: Virus

        Note:
        - u and v are floats or arrays of same shape
        """

        w = - u * v ** 2 + self.F * (1.0 - u)
        return w

    def f_2(self, u, v):
        """Dynamics of the virus.

        Inputs:
        - u: Food
        - v: Virus

        Note:
        - u and v are floats or arrays of same shape
        """

        w = u * v ** 2 - (self.F + self.k) * v
        return w

class Matrix:
    """Matrices used in the project"""

    def A_Lap_1D(self, J):
        """Produces an array modelling 1D Laplace operator with periodic conditions.

        Inputs
        - J: Int - Numbers of samplings of the interval. J+1 points of discretization.

        Return:
        - Array of shape (J+1, J+1)."""

        A = - 2 * np.eye(J + 1) + np.diag(np.ones(J), 1) + np.diag(np.ones(J), -1)
        A[0, -1], A[-1, 0] = 1, 1
        return A

    def A_Lap_2D(self, Jx, Jy, Delta_x , Delta_y):
        """Produces an array modelling 2D Laplace operator on rectangle with periodic conditions.

        Inputs:
        - Jx: Int - Number of sampling of the interval w.r.t. x.
        - Jy: Int - Number of sampling of the interval w.r.t. y.
        - Delta_x: Int - Step size w.r.t. x.
        - Delta_y: Int - Step size w.r.t. y.

        Return:
        - Array of shape ((Jx+1)(Jy+1), (Jx+1)(Jy+1))."""

        Dx = (1 / Delta_x ** 2) * np.kron(np.eye(Jy + 1), self.A_Lap_1D(Jx))
        Dy = (1 / Delta_y ** 2) * np.kron(self.A_Lap_1D(Jy), np.eye(Jx + 1))
        return Dx + Dy

class Operators:
    """Operators used in the project"""

    def Laplace_operator(self, X, Delta_x, Delta_y):
        """Produces an array modelling 2D Laplace operator on rectangle with periodic conditions.

        Inputs:
        - X: 2D Array of shape.
        - Delta_x: Int - Step size w.r.t. x.
        - Delta_y: Int - Step size w.r.t. y.

        Return:
        - Array of same shape of X."""

        Dx = (np.roll(X, shift=(0, 1), axis=(0, 1)) - 2 * X + np.roll(X, shift=(0, -1), axis=(0, 1))) / Delta_x ** 2
        Dy = (np.roll(X, shift=(1, 0), axis=(0, 1)) - 2 * X + np.roll(X, shift=(-1, 0), axis=(0, 1))) / Delta_y ** 2

        return Dx + Dy

class Print_Solution:
    """Class for printing solution"""

    @staticmethod
    def print_PDE_Solution_2D(V, h_params):
        """Prints 2D solution of PDE (V) w.r.t. time.

        Inputs
        - V: Array of shape (N+1, Jx+1, Jy+1) - Approximation of PDE solution.
        - h_params: dict - Hyperparameters of the problem.
        """

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        im = ax.imshow(V[0], origin="lower",
            extent=[0, 1 * h_params['Lx'], 0, h_params['Ly']],
            vmin=np.min(V),
            vmax=np.max(V),
            cmap="jet")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("t = 0")

        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, "t", 0, V.shape[0] - 1, valinit=0, valstep=1)

        def update(val):
            n = int(slider.val)
            t = str(np.round(n * h_params['T'] / h_params['N'], 2))
            im.set_data(V[n])
            ax.set_title(f"t = {t}")
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
        return None

    @staticmethod
    def print_Computed_Solution_2D(sol_path):
        """Prints 2D solution of PDE (V) w.r.t. time after computation and save.

        Inputs
        - sol_path: Str - Path of PDE solution.
        """

        SOL = np.load(sol_path, allow_pickle=True)
        V = SOL["solution"]
        params = SOL["parameters"].item()  # souvent nécessaire si dict

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        im = ax.imshow(V[0], origin="lower",
            extent=[0, 1 * params['Lx'], 0, params['Ly']],
            vmin=np.min(V),
            vmax=np.max(V),
            cmap="jet")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("t = 0")

        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, "t", 0, V.shape[0] - 1, valinit=0, valstep=1)

        def update(val):
            n = int(slider.val)
            t = str(np.round(n * params['T'] / params['N'], 2))
            im.set_data(V[n])
            ax.set_title(f"t = {t}")
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
        plt.close() # Closes the figure from memory
        SOL.close() # Closes the solution from memory
        return None

    @staticmethod
    def save_Computed_Solution_2D(sol_path, sampling = 50):
        """Prints 2D solution of PDE (V) w.r.t. time after computation and save.

        Inputs
        - sol_path: Str - Path of PDE solution.
        - sampling: Int - Sampling interval. For instance, if sampling = 50 and there are 10.000 frames, 200 frames are saved, 1 for 50. Default: 50.
        """
        SOL = np.load(sol_path, allow_pickle=True)
        V = SOL["solution"]
        params = SOL["parameters"].item()

        N = params['N']
        sol_file = os.path.dirname(sol_path)

        SPL = sampling
        for n in range(N // SPL + 1):
            print(" > " + str(n + 1) + " / " + str(N // SPL + 1), end="\r")
            plt.figure()
            plt.imshow(V[int(n * SPL), : , :], cmap="jet", vmin=np.min(V), vmax=np.max(V))
            plt.colorbar()
            plt.title("t = " + str(np.round(SPL * n * params['T'] / params['N'], 2)))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(sol_file + "/PDE_" + str(int(SPL * n)) + ".png", dpi=300)
            plt.close()

        return None