import numpy as np
import math
import matplotlib.pyplot as plt
from tools import *

def main():

    # Givens 
    L = 1                   # Length
    lam = 2                 # Convection Coefficient
    mu = .2                 # Diffusion Coefficient
    ti = 0                  # initial time
    tf = .2                 # final time
    delta_t = .01           # Time step
    N = int(tf/delta_t)     # Number of time steps

    # Discretized Domain
    t = np.linspace(0, .2, num=N)
    x = np.linspace(0, 1, num=11)
    Nx = x.size()
    dx = L/Nx
    U = np.zeros((int(N),11))

    # Inital Conitions
    U[0,:] = -5*pow(x,2) + 5
    print(U)
    
    ## Solve PDE
    # Construct A Matrix
    A = np.zeros((Nx, Nx))

    for i in range(Nx):
        if i != 0 & i != Nx-1:
            A[i,i] = 1/delta_t + 2*mu/pow(dx)
            A[i, i-1] = -(lam/(2*dx) + mu/pow(dx,2))
            A[i, i+1] = lam/(2*dx) - mu/pow(dx,2)
        elif i == 0:
            A[i,i] = 1
        else: 
            A[i,8] = 1
            A[i,9] = -2

    # Solve System

    # Plot at t = 0.1
    plt.plot(x, U[-1,:], label = 'Solution at tf = 0.1')
    plt.xlabel('x (postion)')
    plt.ylabel('U (velocity)')
    plt.title('Solution at t = 0.1')
    plt.show()
main()