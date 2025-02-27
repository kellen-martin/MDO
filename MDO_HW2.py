import numpy as np
import math
import matplotlib.pyplot as plt

def simPDE(U0):

    # Givens 
    L = 1                   # Length
    lam = 2                 # Convection Coefficient
    mu = .2                 # Diffusion Coefficient
    ti = 0                  # initial time
    tf = .1                 # final time
    delta_t = .01           # Time step
    N = int(tf/delta_t)     # Number of time steps

    # Discretized Domain
    ts = np.linspace(0,.1,num=N+1)
    x = np.linspace(0, 1, num=11)
    Nx = 11
    dx = L/(Nx-1)
    U = np.zeros((int(N+1),11))

    # Inital Conitions
    U[0,:] = U0
    
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
    
    return U[-1,:]

def objFunc(U0, V):
    # The mean square error
    U_f = simPDE(U0)
    F = np.sum(np.square(U_f - V))
    return F

# def calcGrad(x, xf):
    # initialize gradiant vector
    dfdx = np.zeros_like(x)

    # FD step size 
    delta = .01

    # Central FD to calculate gradiant
    for i in range(len(x)):
       # Perturb x
        x_plus = x
        x_minus = x
        x_plus[i] = x_plus[i] + delta
        x_minus[i] = x_minus[i] - delta

        # Calculate perturbed function
        F_plus = objFunc(x_plus, xf)
        F_minus = objFunc(x_minus, xf)

        # Calculate Gradiant
        dfdx[i] = (F_plus - F_minus)/(2*delta)

    return dfdx

def calcGrad(x, xf):
    dfdx = np.zeros_like(x)
    eps = 1e-5
    # f0 = calcObj(x)
    print(x)
    for i in range(len(x)):
        x[i] = x[i] + eps
        f1 = objFunc(x, xf)
        x[i] = x[i] - 2.0 * eps
        f2 = objFunc(x, xf)
        dfdx[i] = (f1 - f2) / (2.0 * eps)
        # reset the purb
        x[i] = x[i] + eps
    # FD implementation
    return dfdx

def lineSearch(x0, xf, d, alpha0, rho):
    alpha = alpha0
    f0 = objFunc(x0, xf)

    for i in range(20):
        x = x0 + alpha * d
        f = objFunc(x, xf)
        if f < f0:
            break
        else:
            alpha = alpha * rho

    return alpha

def steepest_decent_opt(x0, xf, alpha0, rho, N_max, tol):
    norm = 500
    n = 0
    x = x0
    f = objFunc(x0, xf)
    dfdx = calcGrad(x,xf)
    d = np.zeros_like(dfdx)

    while(f > tol and n<N_max):
        f = objFunc(x, xf)
        dfdx0 = dfdx
        d0 = 0
        dfdx = calcGrad(x, xf)
        norm = np.linalg.norm(dfdx)

        d = -dfdx/norm

        if(n>0):
            alpha = alpha * np.dot(dfdx0, d0) / np.dot(dfdx, d)
        alpha = lineSearch(x, xf, d, alpha0, rho)

        x = x + alpha*d

        n = n+1

        print("iter:", n, "x: ", x, "norm", norm, "f:", f, "alpha0", alpha)

    if(f < tol):
        print('Optimimal Point Found')
    else:
        print('Did not converge')

    return x


def main():
    # Givens
    V = np.array([5.0, 5.009, 5.045, 5.119, 5.231, 5.371, 5.522, 5.670, 5.800, 5.9004, 6.008])
    x = np.linspace(0, 1, num=11)

    #Initial Guess 
    U0 = 5*np.ones_like(V)



    # Initial Opt. conditions
    alpha_0 = 5
    rho = .5
    N_max = 1000
    tol = 1e-4

    # Optimize
    U0_opt = steepest_decent_opt(U0, V, alpha_0, rho, N_max, tol)
    print("U0 = ", U0_opt)
    Uf = simPDE(U0_opt)
    print("Uf = ", Uf)
    F = objFunc(U0, V)
    print("Objective Function at Optimum Point: ", F)
    # Plot at Initial Time
    plt.plot(x, U0_opt, label = 'Optimal Initial Conditions')
    plt.xlabel('x (postion)')
    plt.ylabel('U (velocity)')
    plt.title('Optimal Initial Conditions')
    plt.ylim([0,7])
    plt.show()

    # Plot Desired vs Found at final time
    plt.plot(x, Uf, "r-", label = 'Optimal Solution')
    plt.plot(x, V,  "k--", label = 'Target Solution')
    plt.legend()
    plt.xlabel('x (postion)')
    plt.ylabel('U (velocity)')
    plt.title('Target vs. Solution as t=0.1')
    plt.ylim([0,7])
    plt.show()

main()