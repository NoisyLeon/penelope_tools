import database
import matplotlib.pyplot as plt
import numpy as np

dbase   = database.penelopedbase()
dbase.read_r_bins_mat(rmax=800., Nr=401, infname='/home/leon/code/InverseTransformSampling/xray_rbins.mat')


dbase2   = database.penelopedbase()
rdata = dbase2.read_r_bins_mat(rmax=800., Nr=401, infname='/home/leon/code/InverseTransformSampling/xray_rbins.mat')
theta = np.random.rand(rdata.size, 1)*2.*np.pi
dbase2.x = rdata*np.cos(theta); dbase2.y = rdata*np.sin(theta)
dbase2.rbins = None
dbase2.count_r_bins( rmax=800., Nr=401)

ratio = dbase.rbins[0]/dbase2.rbins[0]
ratio = 1.
ax=plt.subplot()
plt.plot(dbase.rArr, dbase.rbins, 'ro', ms=10,label='radius')
plt.plot(dbase2.rArr, dbase2.rbins*ratio, 'bo', ms=5,label='xy')
# plt.plot(dbase.rArr, dbase.rbins_gauss, 'k--',lw=5, ms=10,label='gauss')
# plt.plot(dbase.rArr, dbase.rbins_pre, 'k--',lw=5, ms=10,label='gauss')



plt.ylabel('Number of photons', fontsize=30)
plt.xlabel('Radius (nm)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(loc=0, fontsize=20, numpoints=1)
plt.yscale('log', nonposy='clip')
plt.xlim(0, 400.)
# plt.xlim(0, 100.)
# plt.ylim(5e-3, 0.5)
# plt.ylim(1, 1e4)
plt.show()