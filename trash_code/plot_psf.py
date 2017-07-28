# 
import numpy as np
from scipy import stats
from mayavi import mlab

infname = 'from_Amrita/Au_100nm/I_1nA_E_30keV/psf-test.dat'
inArr = np.loadtxt(infname)

x = inArr[:, 2]; y = inArr[:, 3]; z = inArr[:, 4]
zmin = 0.; zmax = 1.e-5; dz = 1.e-6

xyz = np.vstack([x,y,z])
kde = stats.gaussian_kde(xyz)
density = kde(xyz)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')
pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=0.07)
mlab.axes()
mlab.show()



