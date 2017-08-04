
import xraylib_func
import numpy as np
import matplotlib.pyplot as plt


eArr= np.arange(30000.)/1000. + .001
ax=plt.subplot()
ind     = np.arange(7)
tlst    = 10**ind/1e7
i       = 0
chem    = 'C16H14Cl2'
mu      = xraylib_func.get_mu_np_CP(energy=eArr, chemicalform=chem)

for t in tlst:
    ratio   = np.exp(-mu*t)
    i+=1
    plt.plot(eArr*1000., ratio, '-', lw=2, label=str(t*1e7)+' nm')
    # if i < 5:
    #     plt.plot(eArr*1000., ratio, '-', lw=2, label=str(t*1e7)+' nm')
    # else:
    #     plt.plot(eArr*1000., ratio, '--', lw=2, label=str(t*1e7)+' nm')

# plt.yscale('log', nonposy='clip')
plt.ylabel(r'$e^{{-\mu}{t}}$', fontsize=35)
plt.xlabel('energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(numpoints=1, loc=0, fontsize=20)
plt.ylim(-0.1, 1.1)
plt.xlim(0., 15000.)
plt.title('Attenuation Ratio for Parylene C '+r'$(C_{16}H_{14}Cl_{2})$', fontsize=40)
plt.show()