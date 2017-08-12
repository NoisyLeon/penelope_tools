import database
import matplotlib.pyplot as plt
import numpy as np
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
dbase.count_r_bins( rmax=400., Nr=401)

r   = dbase.rArr[dbase.rArr<75]
y   = dbase.rbins_norm[dbase.rArr<75]
a, b = np.polyfit(r, np.log(y), 1)
# 
# rms = 1e9
# for a in np.arange(1000.):
#     print a
#     for b in np.arange(1000.):
#         ypre = np.exp(-a*r+b)
#         temp = (y-ypre)**2
#         rms_temp = np.sqrt(np.mean(temp))
#         if rms_temp <rms:
#             amin = a; bmin=b
# 
# 
ypre = np.exp(a*r+b)



ax=plt.subplot()
plt.plot(dbase.rArr, dbase.rbins_norm, 'ro', ms=5,label='all')
plt.plot(r, ypre, 'k-',lw=5, ms=10,label='pre')


plt.ylabel('Number of photons', fontsize=30)
plt.xlabel('Radius (nm)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
# plt.legend(loc=0, fontsize=20, numpoints=1)
plt.yscale('log', nonposy='clip')
plt.xlim(0, 400.)
# plt.ylim(5e-3, 0.5)
# plt.ylim(1, 1e4)
plt.show()

