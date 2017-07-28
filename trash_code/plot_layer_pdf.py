import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.neighbors import KernelDensity

infname = 'from_Amrita/Au_100nm/I_1nA_E_30keV/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]; y = inArr[:, 3]; z = inArr[:, 4]

# ind_valid = (x >= -1e-5)*(x <= 1e-5) * (y >= -1e-5)*(y <= 1e-5) * (z >= 0.)*(z <= 1e-5)
ind_valid = (z >= 0.)*(z <= 1e-5)
xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]

xyz = np.vstack([xin,yin,zin])

# do kernel density estimation to get smooth estimate of distribution
# make grid of points

N=21j
z0=1e-6
# print 'comuting PDF'
# kernel      = scipy.stats.gaussian_kde(xyz, bw_method=None)
# x, y, z     = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N, z0:z0:1j]
# positions   = np.vstack((x.ravel(), y.ravel(), z.ravel()))
# density     = np.reshape(kernel(positions).T, x.shape)
# print 'End comuting PDF'
# 
# # plot points
# # plot projection of density onto z-axis
# zpdf = np.sum(density, axis=2)
# zpdf = zpdf / zpdf.sum()*100.
# plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
# 
# plt.pcolormesh(plotx, ploty, zpdf, shading='gouraud')
# plt.title('')
# plt.xlabel('X (cm)', fontsize=30)
# plt.ylabel('Y (cm)', fontsize=30)
# plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'equal')
# cb=plt.colorbar()
# cb.set_label('%', fontsize=25)
# plt.show()



kde = KernelDensity().fit(xyz)
x, y, z     = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N, z0:z0:1j]
xyz_plot = np.vstack([xin,yin,zin])
log_dens=kde.score_samples(xyz_plot)
zpdf = np.sum(log_dens, axis=2)

