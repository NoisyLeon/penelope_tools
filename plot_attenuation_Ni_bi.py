
import xraylib_func
import numpy as np
import matplotlib.pyplot as plt

t   = 1.e5
t   = t/1.e7
eArr= np.arange(15000.)/1000. + .001
ax=plt.subplot()
elemlst = ['Ni', 'Bi']
i=0
for elem in elemlst:
    mu  = (xraylib_func.get_mu_np(energy=eArr, elesym=elem))[0,:]
    # ratio= np.exp(-mu*t)
    i+=1
    if i==1:
        t=1.e4/1.e7
        ratio= np.exp(-mu*t)
        plt.plot(eArr*1000., ratio, '-', lw=2, label=elem)
        # break
    else:
        t=4.1e3/1.e7
        ratio= np.exp(-mu*t)
        plt.plot(eArr*1000., ratio, '--', lw=2, label=elem)

# plt.yscale('log', nonposy='clip')
# plt.ylabel(r'$\mu (cm^{-1})$', fontsize=30)
plt.ylabel(r'$e^{{-\mu}{t}}$', fontsize=35)
plt.xlabel('energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(numpoints=1, loc=0, fontsize=20)
# plt.title('Attenuation Ratio for '+elem, fontsize=40)
plt.xlim(0., 15000.)
plt.show()