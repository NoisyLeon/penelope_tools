import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


infname = 'from_Amrita/Au_100nm/I_1nA_E_30keV/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]; y = inArr[:, 3]; z = inArr[:, 4]

ind_valid = (x >= -1e-5)*(x <= 1e-5) * (y >= -1e-5)*(y <= 1e-5) * (z >= 0.)*(z <= 1e-5)
xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]

xyz = np.vstack([xin,yin,zin])

# do kernel density estimation to get smooth estimate of distribution
# make grid of points 
print 'comuting PDF'
kernel      = scipy.stats.gaussian_kde(xyz)
x, y, z     = np.mgrid[-1e-5:1e-5:21j, -1e-5:1e-5:21j, 0:1e-5:11j]
positions   = np.vstack((x.ravel(), y.ravel(), z.ravel()))
density     = np.reshape(kernel(positions).T, x.shape)
print 'End comuting PDF'
# 
# # 
# # # now density is 100x100x100 ndarray
# # 
# plot points
ax = plt.subplot(projection='3d')
# ax.plot(xin, yin, zin, 'o', ms=2)

# plot projection of density onto z-axis
zpdf = np.sum(density, axis=2)
zpdf = zpdf / np.max(zpdf)
plotx, ploty = np.mgrid[-1e-5:1e-5:21j, -1e-5:1e-5:21j]
ax.contour(plotx, ploty, zpdf, offset=0, zdir='z')
# # 
# # #This is new
# # #plot projection of density onto y-axis
# # plotdat = np.sum(density, axis=1) #summing up density along y-axis
# # plotdat = plotdat / np.max(plotdat)
# # plotx, plotz = np.mgrid[-4:4:100j, -4:4:100j]
# # ax.contour(plotx, plotdat, plotz, offset=4, zdir='y')
# # 
# # #plot projection of density onto x-axis
# # plotdat = np.sum(density, axis=0) #summing up density along z-axis
# # plotdat = plotdat / np.max(plotdat)
# # ploty, plotz = np.mgrid[-4:4:100j, -4:4:100j]
# # ax.contour(plotdat, ploty, plotz, offset=-4, zdir='x')
# # #continue with your code
# # 
ax.set_xlim((-1e-5, 1e-5))
ax.set_ylim((-1e-5,1e-5))
ax.set_zlim((0, 1e-5))
# 
# 
