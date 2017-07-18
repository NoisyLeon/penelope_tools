import numpy as np

# input psf file name
infname = 'from_Amrita/Au_100nm/I_10nA_E_30keV/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]; y = inArr[:, 3]; z = inArr[:, 4]

Ntotal = x.size
ind_valid = (x >= -1e-5)*(x <= 1e-5) * (y >= -1e-5)*(y <= 1e-5) * (z >= 0.)*(z <= 1e-5)

ind_valid = (z >= 0.)*(z <= 1e-5)
xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]

Nin = xin.size

print 'Total number =', Ntotal, ' Number of points inside the box =', Nin
