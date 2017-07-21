import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal

pfx = 'I_100nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]; y = inArr[:, 3]; z = inArr[:, 4]

# ind_valid = (x >= -1e-5)*(x <= 1e-5) * (y >= -1e-5)*(y <= 1e-5) * (z >= 0.)*(z <= 1e-5)
ind_valid = (z >= 0.)*(z <= 1e-5)
xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]

xyz     = np.vstack([xin,yin,zin])
COV     = np.cov(xyz)
MEAN    = np.mean(xyz, axis=1)
# do kernel density estimation to get smooth estimate of distribution
# make grid of points

N=201j
z0=10e-6
print 'comuting PDF'
kernel      = scipy.stats.gaussian_kde(xyz, bw_method=None)
x, y, z     = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N, z0:z0:1j]
positions   = np.vstack((x.ravel(), y.ravel(), z.ravel()))
density     = np.reshape(kernel.pdf(positions).T, x.shape)
print 'End comuting PDF'

# 
# plot points
# plot projection of density onto z-axis
zpdf = np.sum(density, axis=2)
zpdf = zpdf / zpdf.sum()*100.
plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
#
fig    = plt.figure(figsize=(12,8))
ax = plt.subplot(221)
plt.pcolormesh(plotx, ploty, zpdf, shading='gouraud', vmax=0.025, vmin=0.)
plt.xlabel('X (cm)', fontsize=10)
plt.ylabel('Y (cm)', fontsize=10)
plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'scaled')
cb=plt.colorbar()
cb.set_label('%', fontsize=10)
plt.title('CDF from KDE')


x, y, z = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N, z0:z0:1j]
pos     = np.dstack((x, y, z))
COV = np.diag(kernel.covariance.diagonal())

rv      = multivariate_normal(mean=[0, 0, 5e-6], cov=COV)
ax = plt.subplot(222)
zpdf2   = rv.pdf(pos)
zpdf2   = zpdf2 / zpdf2.sum()*100.
plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
plt.pcolormesh(plotx, ploty, zpdf2, shading='gouraud', vmax=0.025, vmin=0.)

plt.xlabel('X (cm)', fontsize=10)
plt.ylabel('Y (cm)', fontsize=10)
plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'scaled')
cb=plt.colorbar()
cb.set_label('%', fontsize=10)
plt.title('CDF from best fitting Gaussian distribution')

diff_zpdf = (zpdf - zpdf2)/zpdf.max()

ax = plt.subplot(223)
plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
vmax=np.abs(diff_zpdf).max()
plt.pcolormesh(plotx, ploty, diff_zpdf, shading='gouraud', cmap='seismic', vmax=vmax, vmin=-vmax)

plt.title('')
plt.xlabel('X (cm)', fontsize=10)
plt.ylabel('Y (cm)', fontsize=10)
plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'equal')
cb=plt.colorbar()
# cb.set_label('%', fontsize=25)

plt.suptitle(pfx + ' z = ' +str(z0)+' cm', fontsize=20)

plt.show()



