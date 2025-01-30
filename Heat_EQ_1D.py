import numpy as np
import math
import matplotlib.pyplot as plt
from tools import *

# In class programing section on solving the 1-D Heat equation

def main():
    # user input
    L = 3.0
    dx = 1.0
    time = 2.0
    dt = 1.0
    Nx = int(L/dx) + 1
    Nt = int(time/dt)

    # Initial Condiations 
    T = np.ones(Nx)
    print(T)
    
    # Conctruct A Matrix
    A = np.zeros([Nx, Nx])
    b = np.zeros([Nx, 1])

    x = np.linspace(0, L, Nx)

    # assign values to A matrix
    A[0,0] = 1
    A[-1,-1] = 1

    for i in range(Nx):
        if i == 0:
            A[i,i] = 1.0
        elif i ==Nx - 1:
            A[i,i] = 1
        else:
            A[i,i] = 1/dt + 2/pow(dx,2)
            A[i,i-1] = -1/pow(dx,2)
            A[i,i+1] = -1/pow(dx,2)

    # Time loop
    for t in range(Nt):
        for i in range(Nx):
            if i == 0:
                b[i] = 0
            elif i == Nx - 1:
                b[i] = 1
            else:
                b[i] = T[i]/dt

    print(b)
    # Solve System
    T = np.linalg.solve(A,b)
  

    # Plot at t
    plt.plot(x, T, label = "Time%f" % t)
    plt.xlabel('x (postion)')
    plt.ylabel('T (Temperature')
    plt.title('Solution at t = 2')
    plt.xlim([0, L])
    plt.ylim([0, 5])
    plt.show()


main()