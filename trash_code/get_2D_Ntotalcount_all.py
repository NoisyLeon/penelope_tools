import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
import numba

def count_points(nArr, xg, yg, xin, yin, dx):
    for ix in xrange(xg.size):
        for iy in xrange(yg.size):
            x   = xg[ix]; y = yg[iy]
            xmin = x - dx; xmax = x + dx
            ymin = y - dx; ymax = y + dx
            N=np.where((xin>=xmin)*(xin<xmax)*(yin>=ymin)*(yin<ymax))[0].size
            nArr[ix, iy] = N
    return nArr


plst = ['I_1nA_E_30keV','I_5nA_E_30keV',  'I_10nA_E_30keV', 'I_100nA_E_30keV']

# plst = ['I_1nA_E_30keV']
ax=plt.subplot()
clst=['k', 'b', 'r', 'g']
i=0
for pfx in plst:
    # pfx = 'I_1nA_E_30keV'
    zArr    = np.arange(19)*5.
    nArr    = np.array([])
    for zmin in zArr:
        infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
        inArr = np.loadtxt(infname)
        x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
        
        zmax = zmin + 10
        ind_valid = (z >= zmin)*(z <= zmax)
        xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]
        nArr = np.append(nArr, xin.size)
        print zmin
    zArr    = zArr +5.
    plt.plot(zArr, nArr, clst[i]+'o--', lw=3, ms=8, label=pfx)
    i+=1
plt.legend(loc=0, fontsize=20)
plt.ylabel('Number of points', fontsize=30)
plt.xlabel('Z (nm)', fontsize=30)
# plt.title('1 sigma (68%)')
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.xlim(0., 100.)
plt.show()


