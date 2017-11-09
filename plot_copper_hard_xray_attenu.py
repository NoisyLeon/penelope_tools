
import xraylib_func
import numpy as np
import matplotlib.pyplot as plt

t   = 1.e5
t   = t/1.e7
eArr= np.arange(15000.)/1000. + .001

elemlst = ['Cu']
i=0



e2Arr    = np.array([8495, 9628, 9713, 11442, 11587, 13422])/1000.


mu  = (xraylib_func.get_mu_np(energy=eArr, elesym='Cu'))[0,:]
mu2  = (xraylib_func.get_mu_np(energy=e2Arr, elesym='Cu'))[0,:]


ax=plt.subplot()
plt.plot(eArr*1000., mu, '-', lw=3)
plt.plot(e2Arr*1000., mu2, 'ro', ms=10, lw=2)

# n   = (1000.*e2Arr).tolist()
# for i, txt in enumerate(n):
#     ax.annotate(txt, (e2Arr[i]*1000.,mu2[i]))

plt.yscale('log', nonposy='clip')
plt.ylabel(r'$\mu (cm^{-1})$', fontsize=30)
plt.xlabel('energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(numpoints=1, loc=0, fontsize=20)
plt.title('Attenuation coefficient for Copper', fontsize=40)
plt.xlim(0., 15000.)
plt.show()