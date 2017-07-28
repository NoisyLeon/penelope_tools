import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional


infname = 'from_Amrita/Au_100nm/I_1nA_E_30keV/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]; y = inArr[:, 3]; z = inArr[:, 4]

# ind_valid = (x >= -1e-5)*(x <= 1e-5) * (y >= -1e-5)*(y <= 1e-5) * (z >= 0.)*(z <= 1e-5)
ind_valid = (z >= 0.)*(z <= 1e-5)
xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]

N=21j
z0=1e-6

x, y, z     = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N, z0:z0:1j]
x=x.reshape(x.size); y=y.reshape(x.size); z=z.reshape(x.size)
kde = KDEMultivariateConditional(endog=[xin, yin, zin], exog=[x, y, z], dep_type='u', indep_type='c', bw='normal_reference')


# 
# xyz     = np.vstack([xin,yin,zin])
# COV     = np.cov(xyz)
# MEAN    = np.mean(xyz, axis=1)
# # do kernel density estimation to get smooth estimate of distribution
# # make grid of points
# 
# N=21j
# z0=1e-6
# print 'comuting PDF'
# kernel      = scipy.stats.gaussian_kde(xyz, bw_method=None)
# x, y, z     = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N, z0:z0:1j]
# positions   = np.vstack((x.ravel(), y.ravel(), z.ravel()))
# density     = np.reshape(kernel.pdf(positions).T, x.shape)
# print 'End comuting PDF'
# 
# # 
# # plot points
# # plot projection of density onto z-axis
# zpdf = np.sum(density, axis=2)
# # zpdf = zpdf / zpdf.sum()*100.
# plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
# # 
# plt.pcolormesh(plotx, ploty, zpdf, shading='gouraud')
# plt.title('')
# plt.xlabel('X (cm)', fontsize=30)
# plt.ylabel('Y (cm)', fontsize=30)
# plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'equal')
# cb=plt.colorbar()
# cb.set_label('%', fontsize=25)
# 
# 
# x, y, z = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N, z0:z0:1j]
# pos     = np.dstack((x, y, z))
# rv      = multivariate_normal(mean=MEAN, cov=kernel.covariance)
# fig2    = plt.figure()
# zpdf2   = rv.pdf(pos)
# # zpdf2   = zpdf2 / zpdf2.sum()*100.
# plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
# plt.pcolormesh(plotx, ploty, zpdf2, shading='gouraud')
# 
# plt.title('')
# plt.xlabel('X (cm)', fontsize=30)
# plt.ylabel('Y (cm)', fontsize=30)
# plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'equal')
# cb=plt.colorbar()
# cb.set_label('%', fontsize=25)
# 
# diff_zpdf = (zpdf - zpdf2)/zpdf.max()
# 
# fig3    = plt.figure()
# plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
# plt.pcolormesh(plotx, ploty, diff_zpdf, shading='gouraud')
# 
# plt.title('')
# plt.xlabel('X (cm)', fontsize=30)
# plt.ylabel('Y (cm)', fontsize=30)
# plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'equal')
# cb=plt.colorbar()
# cb.set_label('%', fontsize=25)
# 
# 
# plt.show()



