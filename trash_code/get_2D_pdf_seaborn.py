import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
import numba
import seaborn as sns

def count_points(nArr, xg, yg, xin, yin, dx):
    for ix in xrange(xg.size):
        for iy in xrange(yg.size):
            x   = xg[ix]; y = yg[iy]
            xmin = x - dx; xmax = x + dx
            ymin = y - dx; ymax = y + dx
            N=np.where((xin>=xmin)*(xin<xmax)*(yin>=ymin)*(yin<ymax))[0].size
            nArr[ix, iy] = N
    return nArr


pfx = 'I_1nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7

zmin = 0; zmax = zmin + 10
ind_valid = (z >= zmin)*(z <= zmax)
xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]


# sns.kdeplot(xin, yin, shade=True, cut=5)
# g = sns.jointplot(xin, yin, kind="scatter", size=7, space=0)
g = sns.jointplot(xin, yin, kind="kde", size=7, space=0, xlim={-800, 800}, ylim={-800, 800})
# 
# N       = 201j
# xg      = np.mgrid[-100:100:N]
# yg      = np.mgrid[-100:100:N]
# nArr    = np.zeros((xg.size, yg.size))
# 
# nArr    = count_points(nArr, xg, yg, xin, yin, dx=(xg[1]-xg[0])/2.)
# Nt      = xin.size
# rArr    = np.arange(1000.)*0.1+0.1
# plotx, ploty = np.mgrid[-100:100:N, -100:100:N]
# 
# for r in rArr:
#     # print (nArr[(plotx**2+ploty**2)<=r**2]).sum(), Nt
#     if (nArr[(plotx**2+ploty**2)<=r**2]).sum() >= Nt*0.95:
#         print 'half radius = ',r,' nm', ' Nin = ', nArr[(plotx**2+ploty**2)<=r**2].sum(), 'Ntotal =', Nt
#         break
# 
# pdf     = nArr/Nt
# xpdf    = pdf[100, :]
# ypdf    = pdf[:, 100]
# 
# 
# plotx, ploty = np.mgrid[-100:100:N, -100:100:N]
# 
# # #
# fig     = plt.figure(figsize=(12,8))
# ax      = plt.subplot(221)
# plt.pcolormesh(plotx, ploty, pdf, shading='gouraud', vmax=pdf.max(), vmin=0.)
# plt.xlabel('X (nm)', fontsize=10)
# plt.ylabel('Y (nm)', fontsize=10)
# plt.axis([-100, 100, -100, 100], 'scaled')
# cb=plt.colorbar()
# cb.set_label('%', fontsize=10)
# 
# 
# ax = plt.subplot(222)
# plt.text(0.2, 0.5, 'half radius = '+str(r)+' nm', fontsize=15)
# 
# ax = plt.subplot(223)
# plt.plot(xg, nArr[100,:], 'r-', lw=3, label='PDF, y = 0 nm')
# plt.plot(yg, nArr[:, 100], 'k--', lw=3, label='PDF, x = 0 nm')
# 
# ax = plt.subplot(224)
# # 
# plt.plot(xg, xpdf, 'r-', lw=3, label='PDF, y = 0 nm')
# plt.plot(yg, ypdf, 'k--', lw=3, label='PDF, x = 0 nm')
# 
# 
# 
# plt.xlim(-100, 100)
# plt.legend(loc=0, fontsize=10)
plt.show()



