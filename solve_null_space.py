
import xraylib_func
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import random

eArr    = 8200. + np.arange(50.) * 106.
# eArr    = np.array([8495., 9628, 9711, 11443, 11587, 13423])
eArr    /= 1000.

# eArr    = np.array([8495, 9628, 9713, 11442, 11587, 13422])/1000.
elemlst = ['Hf', 'Ta', 'W', 'Bi', 'Au', 'Pb', 'Zn', 'Cu', 'Pt', 'Si', 'Al']
# elemlst = ['Al', 'Hf', 'Ta', 'W', 'Bi', 'Au', 'Pb', 'Zn', 'Cu', 'Pt', 'Si']

# random.shuffle(elemlst)

Ne      = len(elemlst)

M       = np.zeros([eArr.size, Ne])
M2      = np.zeros([eArr.size, Ne-1])

misfit  = np.zeros(Ne)
CMat    = np.zeros([11, 11])
for j in xrange(11):
    i           = 0
    k           = 0
    for elem in elemlst:
        mu      = (xraylib_func.get_mu_np(energy=eArr, elesym=elem))[0,:]
        M[:, i] = mu[:]
        if i != j:
            M2[:, k]= mu[:]
            k       +=1
        i       += 1
    
    alpha       = np.zeros(Ne)/Ne
    alpha[j]    = 1.
    data        = np.dot(M, alpha)
    
    ######################################
    # # data_noise    =data+ (data*0.001) * np.random.randn(data.size)
    data_noise  = data
    N           = 1
    M3          = np.zeros([eArr.size+N, Ne-1])
    # # # for i in xrange(Ne-1):
    M3[:-N, :]  = M2[:, :]
    # 
    M3[-N:, :]  = 1e5
    data2       = np.zeros(eArr.size+N)
    data2[:-N]  = data_noise
    data2[-N:]  = 1e5
    # 
    # # x_noissfree=np.linalg.lstsq(M2, data2)[0]
    # 
    res         = lsq_linear(M3, data2, bounds=(0., 1.))
    x_noisy     = res.x
    data_pre    = np.dot(M3, x_noisy)
    misfit[j]   = np.sqrt( ( (data - data_pre[:-1])**2/(data**2) ).sum() /data.size)
    
    CMat[j, :j]     = x_noisy[:j]
    CMat[j, j+1:]   = x_noisy[j:]
    
# ################################
# data_noise    =data+ (data*0.01) * np.random.randn(data.size)
# N=1
# M2      = np.zeros([eArr.size+N, Ne])
# for i in xrange(Ne):
#     M2[:-N, i]    = M[:, i]
# 
# M2[-N:, :]  = 1e5
# data2       = np.zeros(eArr.size+N)
# data2[:-N]  = data_noise
# data2[-N:]  = 1e5
# 
# # x_noissfree=np.linalg.lstsq(M2, data2)[0]
# 
# res=lsq_linear(M2, data2, bounds=(0., 1.))
# x_noisy2=res.x
# ################################
# data_noise    =data+ (data*0.05) * np.random.randn(data.size)
# N=1
# M2      = np.zeros([eArr.size+N, Ne])
# for i in xrange(Ne):
#     M2[:-N, i]    = M[:, i]
# 
# M2[-N:, :]  = 1e5
# data2       = np.zeros(eArr.size+N)
# data2[:-N]  = data_noise
# data2[-N:]  = 1e5
# 
# # x_noissfree=np.linalg.lstsq(M2, data2)[0]
# 
# res=lsq_linear(M2, data2, bounds=(0., 1.))
# x_noisy3=res.x
# 
#
ax=plt.subplot()

im  = plt.pcolormesh(CMat.T, cmap='inferno_r', vmin=0., vmax=1.)
plt.xticks(np.arange(11)+0.5)
plt.yticks(np.arange(11)+0.5)
plt.ylabel('basis element', fontsize=20)
plt.xlabel('target element', fontsize=20)
plt.title('Hard X-ray and Bremsstrahlung representation ', fontsize=30)
ax.set_xticklabels(elemlst)
ax.tick_params(axis='x', labelsize=20)
ax.set_yticklabels(elemlst)
ax.tick_params(axis='y', labelsize=20)

# for tick in ax.xaxis.get_major_ticks():
#     tick.tick1line.set_markersize(0)
#     tick.tick2line.set_markersize(0)
#     tick.label1.set_horizontalalignment('center')
    
    
plt.colorbar(im)
plt.xlim(0, 11)
plt.ylim(0, 11)
plt.show()
# 
ax=plt.subplot()
plt.plot(misfit, 'bo-.', ms= 15)

plt.xticks(np.arange(11))
# plt.title('V', fontsize=40)
ax.set_xticklabels(elemlst)
plt.yscale('log', nonposy='clip')

# plt.legend(fontsize=15)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.ylabel('misfit', fontsize=30)
plt.title('Hard X-ray representation ', fontsize=40)
plt.show()