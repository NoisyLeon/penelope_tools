import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
import numba




plst = ['I_1nA_E_30keV','I_5nA_E_30keV',  'I_10nA_E_30keV', 'I_100nA_E_30keV']

# plst = ['I_1nA_E_30keV']
ax=plt.subplot()
clst=['k', 'b', 'r', 'g']
i=0
for pfx in plst:
    # pfx = 'I_1nA_E_30keV'
    dArr    = np.array([])
    zArr    = np.arange(19)*5.
    for zmin in zArr:
        infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
        inArr = np.loadtxt(infname)
        x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
        
        zmax = zmin + 10
        ind_valid = (z >= zmin)*(z <= zmax)
        xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]
        rArr    = np.arange(2000.)*0.05+0.05
        z0      = (zmin+zmax)/2
        Nt      = xin.size
        for r in rArr:
            if (xin[(xin**2+yin**2)<=r**2]).size > Nt*0.5:
                print 'z0 =',z0,' nm 1 sigma radius = ',r,' nm', ' Nin = ', (xin[(xin**2+yin**2)<=r**2]).size, 'Ntotal =', Nt
                break
        
#     
        dArr    = np.append(dArr, 2.*r)
    zArr    = zArr +5.
    plt.plot(zArr, dArr, clst[i]+'o--', lw=3, ms=8, label=pfx)
    i+=1
plt.legend(loc=0, fontsize=20)
plt.ylabel('Characteristic Diameter', fontsize=30)
plt.xlabel('Z (nm)', fontsize=30)
# plt.title('1 sigma (68 %)', fontsize=40)
plt.title('half (50 %)', fontsize=40)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.xlim(0., 100.)
plt.show()


