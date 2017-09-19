import numpy as np
import matplotlib.pyplot as plt
sigma=30.
mean = [0, 0]
cov = [[sigma**2, 0], [0, sigma**2]]  # diagonal covariance
# cov = [[sigma, 0], [0, sigma]]  # diagonal covariance
#Diagonal covariance means that points are oriented along x or y-axis:

N=np.int64(1e7)
x, y = np.random.multivariate_normal(mean, cov, N).T
# plt.plot(x, y, 'o')
# plt.axis('equal')
# plt.show()

R=np.sqrt(x**2+y**2)

print R[R<sigma].size/float(N)

print R[R<2.*sigma].size/float(N)

xg = np.arange(10001.)*.1-500.
yg = np.arange(10001.)*.1-500.
xg, yg = np.meshgrid(xg, yg, indexing='ij')
f = 1./np.pi/2./(sigma**2)*np.exp(-0.5*((xg**2)/(sigma**2))-0.5*((yg**2)/(sigma**2)))

Rg = np.sqrt(xg**2+yg**2)

print (f[Rg<sigma]).sum()*0.1*0.1

print (f[Rg<2*sigma]).sum()*0.1*0.1