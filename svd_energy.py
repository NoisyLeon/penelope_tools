
import xraylib_func
import numpy as np
import matplotlib.pyplot as plt

t   = 1.e5
t   = t/1.e7
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


# i=0
# for elem in elemlst:
#     M2[:, i] = mu[:]
#     i+=1
print np.linalg.matrix_rank(M)
    

U, s, V = np.linalg.svd(M, full_matrices=False)

# s   = s/s.max()

# ax=plt.subplot()
# plt.plot(np.arange(11)+1, s, 'o-', ms=10)
# plt.xticks(np.arange(11)+1)


# plt.plot(np.arange(6)+1, s, 'o-', ms=10)
# plt.xticks(np.arange(6)+1)
# plt.ylabel(r'$\lambda $', fontsize=30)
# plt.xlabel('index', fontsize=30)
# plt.title('Eigenvalues for 50 energy points', fontsize=40)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# plt.yscale('log', nonposy='clip')
# plt.show()

ax=plt.subplot()
R = np.dot( np.linalg.pinv(M, 1e-2), M)
im = plt.pcolormesh(V, cmap='seismic', vmin=-1., vmax=1.)


plt.colorbar(im)
plt.xticks(np.arange(11)+1)
plt.yticks(np.arange(11)+1)
plt.title('V', fontsize=40)
ax.set_xticklabels(elemlst)
ax.set_yticklabels(elemlst)
plt.show()


# a = np.zeros((8, 9))
# a[0, :] = np.array([1,0,0, 1,0,0, 1, 0, 0])
# a[1, :] = np.array([0,1,0, 0,1,0, 0, 1, 0])
# a[2, :] = np.array([0,0,1, 0,0,1, 0, 0, 1])
# a[3, :] = np.array([1,1,1, 0,0,0, 0, 0, 0])
# a[4, :] = np.array([0,0,0, 1,1,1, 0, 0, 0])
# a[5, :] = np.array([0,0,0, 0,0,0, 1,1,1])
# a[6, :] = np.array([np.sqrt(2),0,0, 0,np.sqrt(2),0, 0, 0, np.sqrt(2)])
# a[7, :] = np.array([0,0,0, 0,0,0, 0, 0, np.sqrt(2)])
# 
# # a= np.zeros((10,5))
# # a[0,0] = 1
# # a[1,2] = -1
# U, s, V = np.linalg.svd(a, full_matrices=False)
# im = plt.pcolormesh(V, cmap='seismic', vmin=-.8, vmax=.8)
# plt.colorbar(im)
# plt.show()