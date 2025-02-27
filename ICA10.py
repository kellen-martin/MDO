import numpy as np
import matplotlib.pyplot as plt

# f = (1-x1)^2 + 100*(x2-x1^2)^2


def calcobj(x):
    f = x[0]**3 + 2*x[1]**2 - 4*x[0] - 2*x[0]*x[1]*x[1]

    return f


def calcContraint(x):
    h = x[0] + x[1] - 1
    return h

# Penalty Method
# def calcObj(x):
    rho = 100.0
    F = calcobj(x) + rho*(calcContraint(x)**2)
    return F

# Lagrange Method
def calcObj(x):
    # L = f + lambda*h
    # x[2] is the Lagrange mult
    L = calcobj(x) + x[2]*calcContraint(x)
    return L

def calcHessianFD(x):
    eps = 1e-5
    size_x = len(x)
    H = np.zeros((size_x, size_x))
    for i in range(size_x):
        for j in range(size_x):
            x_ip_jp = x.copy()
            x_ip_jm = x.copy()
            x_im_jp = x.copy()
            x_im_jm = x.copy()

            x_ip_jp[i] += eps
            x_ip_jp[j] += eps

            x_ip_jm[i] += eps
            x_ip_jm[j] -= eps

            x_im_jp[i] -= eps
            x_im_jp[j] += eps

            x_im_jm[i] -= eps
            x_im_jm[j] -= eps

            H[i, j] = (calcObj(x_ip_jp) - calcObj(x_ip_jm) - calcObj(x_im_jp) + calcObj(x_im_jm)) / (4 * eps**2)

    return H


def calcObjGradFD(x):
    dfdx = np.zeros_like(x)
    eps = 1e-5
    # f0 = calcObj(x)
    # print(x)

    for i in range(len(x)):
        x[i] = x[i] + eps
        f1 = calcObj(x)
        x[i] = x[i] - 2.0 * eps
        f2 = calcObj(x)
        dfdx[i] = (f1 - f2) / (2.0 * eps)

        # reset the purb
        x[i] = x[i] + eps

    # FD implementation

    return dfdx


def lineSearch(x0, d, alpha0):
    f0 = calcObj(x0)
    alpha = alpha0

    for i in range(5):
        x1 = x0 + alpha * d
        f1 = calcObj(x1)

        if f1 < f0:
            return alpha
        else:
            alpha = alpha * 0.5

    return alpha


dfdx = np.zeros(2)
d = np.zeros(2)
alpha = 0.0
x = np.array([2.0, 1.0, 2.0])

xAll = []

for i in range(100):
    xAll.append(list(x))

    f = calcObj(x)
    dfdx = calcObjGradFD(x)
    norm = np.linalg.norm(dfdx)
    if norm < 1e-4:
        print("optimal point found", x)
        break
    else:
        H = calcHessianFD(x)
        d = -np.dot(np.linalg.inv(H), dfdx)

        alpha = 1.0

        alpha = lineSearch(x, d, alpha)

        x = x + alpha * d

        print("i: ", i, " f ", f, " norm ", norm)


x0 = np.linspace(-3, 3, 100)
x1 = np.linspace(-3, 3, 100)
X0, X1 = np.meshgrid(x0, x1)
#F = calcObj([X0, X1])
#plt.contour(X0, X1, F, levels=200)

xAll = np.asarray(xAll)
print(xAll[-1,:])

# plt.plot(xAll[:, 0], xAll[:, 1], "-ko")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.ylim([0, 5])
# plt.xlim([0, L])
# plt.legend()
# plt.show()
