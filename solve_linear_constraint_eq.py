
import xraylib_func
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

eArr    = 8200. + np.arange(50.) * 106.
eArr    /= 1000.

# eArr    = np.array([8495, 9628, 9713, 11442, 11587, 13422])/1000.
elemlst = ['Hf', 'Ta', 'W', 'Bi', 'Au', 'Pb', 'Zn', 'Cu', 'Pt', 'Si', 'Al']
# elemlst = ['Al', 'Hf', 'Ta', 'W', 'Bi', 'Au', 'Pb', 'Zn', 'Cu', 'Pt', 'Si']

Ne  = len(elemlst)

M   = np.zeros([eArr.size, Ne])
M2  = np.zeros([eArr.size, Ne])
i=0
for elem in elemlst:

    mu  = (xraylib_func.get_mu_np(energy=eArr, elesym=elem))[0,:]
    M[:, i] = mu[:]
    i   += 1

# alpha = np.zeros(Ne)
# for i in xrange(Ne-1):
#     alpha[i]    = np.random.uniform(0., 1 - alpha.sum())
# alpha[-1]   = 1. -  alpha.sum()

alpha = np.ones(Ne)/Ne


data    = np.dot(M, alpha)
x_noissfree=np.linalg.lstsq(M, data)[0]
######################################
data_noise    =data+ (data*0.001) * np.random.randn(data.size)
N=1
M2      = np.zeros([eArr.size+N, Ne])
for i in xrange(Ne):
    M2[:-N, i]    = M[:, i]

M2[-N:, :]  = 1e5
data2       = np.zeros(eArr.size+N)
data2[:-N]  = data_noise
data2[-N:]  = 1e5

# x_noissfree=np.linalg.lstsq(M2, data2)[0]

res=lsq_linear(M2, data2, bounds=(0., 1.))
x_noisy=res.x
################################
data_noise    =data+ (data*0.01) * np.random.randn(data.size)
N=1
M2      = np.zeros([eArr.size+N, Ne])
for i in xrange(Ne):
    M2[:-N, i]    = M[:, i]

M2[-N:, :]  = 1e5
data2       = np.zeros(eArr.size+N)
data2[:-N]  = data_noise
data2[-N:]  = 1e5

# x_noissfree=np.linalg.lstsq(M2, data2)[0]

res=lsq_linear(M2, data2, bounds=(0., 1.))
x_noisy2=res.x
################################
data_noise    =data+ (data*0.05) * np.random.randn(data.size)
N=1
M2      = np.zeros([eArr.size+N, Ne])
for i in xrange(Ne):
    M2[:-N, i]    = M[:, i]

M2[-N:, :]  = 1e5
data2       = np.zeros(eArr.size+N)
data2[:-N]  = data_noise
data2[-N:]  = 1e5

# x_noissfree=np.linalg.lstsq(M2, data2)[0]

res=lsq_linear(M2, data2, bounds=(0., 1.))
x_noisy3=res.x



ax=plt.subplot()
plt.plot(alpha, 'o', ms= 15, label='real model')
plt.plot(x_noisy, '^', ms= 15, label='reconstructed model, 0.1 % Gausssian noise')
plt.plot(x_noisy2, 'v', ms= 15, label='reconstructed model, 1 % Gausssian noise')
plt.plot(x_noisy3, 'kx', ms= 15, label='reconstructed model, 5 % Gausssian noise')

plt.xticks(np.arange(11))
plt.title('V', fontsize=40)
ax.set_xticklabels(elemlst)
plt.legend(fontsize=15)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.ylabel('element fraction', fontsize=30)
plt.title('Reconstruction of element fraction', fontsize=40)
plt.show()