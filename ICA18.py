import numpy as np
import matplotlib.pyplot as plt

xt = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
nsamples = len(xt)

def GaussianKernel(x1, x2):
    theta = 10.0
    size1 = len(x1)
    size2 = len(x2)
    K = np.zeros((size1, size2))

    for i in range(size1):
        for j in range(size2):
            d2 = (x1[i] - x2[j])**2
            K[i, j] = np.exp(-theta*d2)

    return K

K = GaussianKernel(xt, xt)
Kinv = np.linalg.inv(K)
yt = yt.reshape((5, 1))
KinvY = np.dot(Kinv, yt)

def predict_val(x):
    k = GaussianKernel(xt, [x])
    y = np.dot(k.flatten(), KinvY.flatten())
    return y

y = predict_val(0.3)
print(y)