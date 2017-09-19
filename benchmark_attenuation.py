
import xraylib_func
import numpy as np
import matplotlib.pyplot as plt

elem='Ni'
eArr= np.arange(200000.)*0.1 +0.1
ax=plt.subplot()
muoverrho  = (xraylib_func.get_muoverrho_np(energy=eArr, elesym=elem))[0,:]
plt.plot(eArr, muoverrho, 'k-', lw=2, label=elem+' (Xraylib)')
inArr = np.loadtxt('Ni_muoverrho.txt')
e = inArr[:,0]; data = inArr[:,1]
plt.plot(e*1000., data, 'ro', lw=2, ms=10, label=elem+' (NIST)')

# elem='Bi'
# muoverrho  = (xraylib_func.get_muoverrho_np(energy=eArr, elesym=elem))[0,:]
# plt.plot(eArr/1000., muoverrho, 'b-', lw=2, label=elem+' (Xraylib)')
# inArr = np.loadtxt('Bi_muoverrho.txt')
# e = inArr[:,0]; data = inArr[:,1]
# plt.plot(e, data, 'go', lw=2, ms=5,  label=elem+' (NIST)')


plt.yscale('log', nonposy='clip')
plt.xscale('log', nonposy='clip')
# plt.ylabel(r'$\mu (cm^{-1})$', fontsize=30)
plt.ylabel(r'${\mu}/{\rho}(cm^{2}/g)}$', fontsize=35)
plt.xlabel('Energy (keV)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(numpoints=1, loc=0, fontsize=20)
# plt.title('Attenuation Ratio for '+elem, fontsize=40)
plt.xlim(1, 1e2)
plt.show()