import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# generate some points of a 3D Gaussian
points = np.random.normal(size=(3, 50))

# do kernel density estimation to get smooth estimate of distribution
# make grid of points
x, y, z = np.mgrid[-4:4:100j, -4:4:100j, -4:4:100j]
kernel = sp.stats.gaussian_kde(points)
positions = np.vstack((x.ravel(), y.ravel(), z.ravel()))
density = np.reshape(kernel(positions).T, x.shape)

# now density is 100x100x100 ndarray

# plot points
ax = plt.subplot(projection='3d')
ax.plot(points[0,:], points[1,:], points[2,:], 'o')

# plot projection of density onto z-axis
plotdat = np.sum(density, axis=2)
plotdat = plotdat / np.max(plotdat)
plotx, ploty = np.mgrid[-4:4:100j, -4:4:100j]
ax.contour(plotx, ploty, plotdat, offset=-4, zdir='z')

#This is new
#plot projection of density onto y-axis
plotdat = np.sum(density, axis=1) #summing up density along y-axis
plotdat = plotdat / np.max(plotdat)
plotx, plotz = np.mgrid[-4:4:100j, -4:4:100j]
ax.contour(plotx, plotdat, plotz, offset=4, zdir='y')

#plot projection of density onto x-axis
plotdat = np.sum(density, axis=0) #summing up density along z-axis
plotdat = plotdat / np.max(plotdat)
ploty, plotz = np.mgrid[-4:4:100j, -4:4:100j]
ax.contour(plotdat, ploty, plotz, offset=-4, zdir='x')
#continue with your code

ax.set_xlim((-4, 4))
ax.set_ylim((-4, 4))
ax.set_zlim((-4, 4))


