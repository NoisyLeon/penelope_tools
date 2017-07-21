
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

pfx = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
# 
# zmin = 0; zmax = zmin + 10
# ind_valid = (z >= zmin)*(z <= zmax)
# xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]
# 

energy = inArr[:, 1]

ax=plt.subplot(211)
# the histogram of the data
n, bins, patches = plt.hist(np.repeat(energy, 6250), bins=7500, normed=False, facecolor='blue', edgecolor='blue')
# n, bins, patches = plt.hist(energy, bins=15000, normed=False, facecolor='blue', edgecolor='blue')

plt.ylabel('Photons/sec', fontsize=30)
plt.xlabel('Energy (eV)', fontsize=30)
# plt.title('2 sigma (95%)', fontsize=40)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
# plt.xlim(0., 100.)

plt.yscale('log', nonposy='clip')
plt.ylim(1e4, 1e8)
dbins=bins[0] - bins[1]
plt.grid(True)


ax=plt.subplot(212)
# the histogram of the data
n, bins, patches = plt.hist(np.repeat(energy, 6250), bins=7500, normed=False, facecolor='red', edgecolor='red')
# n, bins, patches = plt.hist(energy, bins=15000, normed=False, facecolor='red', edgecolor='red')
plt.ylabel('Photons/sec', fontsize=30)
plt.xlabel('Energy (eV)', fontsize=30)
# plt.title('2 sigma (95%)', fontsize=40)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
# plt.xlim(0., 100.)
dbins=bins[1] - bins[0]
plt.grid(True)
plt.suptitle('10 nA, 100 nm Au film, 30 keV, 9.75 nm beam diameter, dE = 4 eV', fontsize=40)
plt.show()

cbins = (bins[1:] + bins[:-1])/2

# 8495 eV, 9628 eV, 9711 eV, 11443 eV, 11587 eV, 13423 eV
dbins = 2*dbins
print n[(cbins > (8495.-dbins))*( cbins<(8495.+dbins) )], cbins[(cbins > (8495.-dbins))*( cbins<(8495.+dbins) )]
print n[(cbins > (9628-dbins))*( cbins<(9628+dbins) )], cbins[(cbins > (9628-dbins))*( cbins<(9628+dbins) )]
print n[(cbins > (9711-dbins))*( cbins<(9711+dbins) )], cbins[(cbins > (9711-dbins))*( cbins<(9711+dbins) )]
print n[(cbins > (11443-dbins))*( cbins<(11443+dbins) )], cbins[(cbins > (11443-dbins))*( cbins<(11443+dbins) )]
print n[(cbins > (11587-dbins))*( cbins<(11587+dbins) )], cbins[(cbins > (11587-dbins))*( cbins<(11587+dbins) )]
print n[(cbins > (13423-dbins))*( cbins<(13423+dbins) )], cbins[(cbins > (13423-dbins))*( cbins<(13423+dbins) )]


