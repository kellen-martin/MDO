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
    ts = np.linspace(0,.2,num=N+1)
    x = np.linspace(0, 1, num=11)
    Nx = 11
    dx = L/(Nx-1)
    U = np.zeros((int(N+1),11))

    # Inital Conitions
    U[0,:] = -5*pow(x,2) + 5
    
    ## Solve PDE
    # Construct A Matrix
    A = np.zeros([Nx, Nx])

    for i in range(Nx-1):
        if i != 0 & i != Nx-1:
            A[i,i] = 1/delta_t + 2*mu/pow(dx,2)
            A[i, i-1] = -(lam/(2*dx) + mu/pow(dx,2))
            A[i, i+1] = lam/(2*dx) - mu/pow(dx,2)
        else:
            A[i,i] = 1

    A[-1,8] = 1
    A[-1,9] = -2
    A[-1,-1] = 1

    # Solve System
    for t in range(N):
       # Create b vector
       b = U[t,:]/delta_t

       # Boundary Conditions
       b[0] = 5
       b[-1] = 0

       # Solve System of Equations
       U[t + 1, :] = np.linalg.solve(A,b)
    
    # Analyze U at t=0.1
    print(U[10,:])
    # Plot at t = 0.1
    plt.plot(x, U[10,:], label = 'Solution at tf = 0.1')
    plt.xlabel('x (postion)')
    plt.ylabel('U (velocity)')
    plt.title('Solution at t = 0.1')
    plt.ylim([0,7])
    plt.show()
main()