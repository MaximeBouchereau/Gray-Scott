import numpy as np
import scipy
import importlib
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

import ToolBox
importlib.reload(ToolBox)

# Hyperparameters
params_0 = {'D_u':0.00002, 'D_v':0.00001, 'F':0.06, 'k':0.062, 'Lx':1, 'Ly':1, 'Jx':100, 'Jy':100, 'T':10000, 'N':10000, 'ID':"GN_0.30"} # Starter pack
params_1 = {'D_u':0.005, 'D_v':0.0025, 'F':0.035, 'k':0.065, 'Lx':50, 'Ly':50, 'Jx':200, 'Jy':200, 'T':10000, 'N':10000, 'ID':"GN_0.1"} # Mitosis Turing
params_2 = {'D_u':0.01, 'D_v':0.005, 'F':0.06, 'k':0.062, 'Lx':50, 'Ly':50, 'Jx':200, 'Jy':200, 'T':10000, 'N':10000, 'ID':"CS"} # Maze Turing
params_3 = {'D_u':0.005, 'D_v':0.0025, 'F':0.025, 'k':0.06, 'Lx':50, 'Ly':50, 'Jx':200, 'Jy':200, 'T':10000, 'N':10000, 'ID':"RS"} # Mitosis Turing
params_4 = {'D_u':0.005, 'D_v':0.0025, 'F':0.06, 'k':0.062, 'Lx':50, 'Ly':50, 'Jx':200, 'Jy':200, 'T':10000, 'N':10000, 'ID':"RS"} # Maze Turing
params_5 = {'D_u':0.005, 'D_v':0.0025, 'F':0.045, 'k':0.065, 'Lx':50, 'Ly':50, 'Jx':200, 'Jy':200, 'T':10000, 'N':10000, 'ID':"RS"} # Bacteria

params = params_1

class Virus_Food_PDE:
    """Class for simulating Virus-Food-PDE."""

    @classmethod
    def Integrate(cls, params):
        """Approximates solution of Virus-Food-PDE.

        Inputs:
        - params: dict - Dictionary of parameter values. Contains:
            > D_u: Float - Diffusion coefficient for u (Food)
            > D_v: Float - Diffusion coefficient for v (Virus)
            > F: Float - Coefficient for food creation
            > k: Float - Coefficient for virus limit (limited life expectency for virus)
            > Lx: Float - Size of domain w.r.t. x
            > Ly: Float - Size of domain w.r.t. y
            > Jx: Int - Number of space discretizations w.r.t. x
            > Jy: Int - Number of space discretizations w.r.t. y
            > T: Float - Length of time interval
            > N: Int - Number of time discretizations
            > ID: Str - Initial distribution for V: way that virus is distributed w.r.t. x and y. Options:
                -> "CS": Central square
                -> "RS": Random squares (10 small random squares in space)
                -> "GN_" + str(sigma): Gaussian noise with specified scale. For instance, if sigma = 0.25, write "GN_0.25".

        Numerical integration scheme is Forward Euler with approximation of Laplace operator on the grid with finite differences, boundary conditions."""

        # Step sizes
        Delta_x = params['Lx'] / params['Jx']
        Delta_y = params['Ly'] / params['Jy']
        Delta_t = params['T'] / params['N']

        # Initialization
        U = np.zeros((params['N'] + 1, params['Jx'] + 1, params['Jy'] + 1))
        V = np.zeros((params['N'] + 1, params['Jx'] + 1, params['Jy'] + 1))
        U[0,:,:] = 1.0

        if params['ID'][:2] == "GN":
            V[0, :, :] = np.random.normal(scale=float(params['ID'][4:]), size = (params['Jx'] + 1, params['Jy'] + 1))
        if params['ID'] == "CS":
            V[0, params['Jx'] // 2 - 10:params['Jx'] // 2 + 10, params['Jy'] // 2 - 10:params['Jy'] // 2 + 10] = 0.25
            V[0, params['Jx']//2-5:params['Jx']//2+5, params['Jy']//2-5:params['Jy']//2+5] = 0.0
        if params['ID'] == "RS":
            idx_x, idx_y = np.random.randint(0, params['Jx']-10, size=50), np.random.randint(0, params['Jy']-10, size=50)
            for j in range(idx_x.shape[0]):
                V[0, idx_x[j]:idx_x[j] + 10, idx_y[j]:idx_y[j] + 10] = 0.5

        for n in range(params['N']):
            print("n = " + str(n+1) + " / " + str(params['N']), end="\r")
            U[n + 1, :, :] = U[n, :, :] + params['D_u'] * Delta_t * ToolBox.Operators().Laplace_operator(U[n, :, :], Delta_x, Delta_y) + Delta_t * ToolBox.Vector_Field(params['k'], params['F']).f_1(U[n, :, :], V[n, :, :])
            V[n + 1, :, :] = V[n, :, :] + params['D_v'] * Delta_t * ToolBox.Operators().Laplace_operator(V[n, :, :], Delta_x, Delta_y) + Delta_t * ToolBox.Vector_Field(params['k'], params['F']).f_2(U[n, :, :], V[n, :, :])



        ToolBox.Print_Solution().print_PDE_Solution_2D(np.transpose(V, axes=(0, 2, 1)), params)
        # ToolBox.Print_Solution().print_PDE_Solution_2D(np.concatenate((np.transpose(U, axes=(0, 2, 1)), np.transpose(V, axes=(0, 2, 1))), axis=2), params)

        return None