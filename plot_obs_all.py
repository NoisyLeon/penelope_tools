import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
ax=plt.subplot()
dbase.count_r_bins( rmax=800., Nr=401)
rArr = dbase.rArr
plt.plot(rArr, dbase.rbins, 'bo', ms=10,label='all')

# 
# dbase.count_r_bins( zmin=0, rmax=800., Nr=401)
# plt.plot(rArr, dbase.rbins, 'go', ms=10, label='z = 5 nm')
# 
# dbase.count_r_bins( zmin=45, rmax=800., Nr=401)
# plt.plot(rArr, dbase.rbins, 'ro', ms=10,label='z = 50 nm')
# 
# dbase.count_r_bins( zmin=90, rmax=800., Nr=401)
# plt.plot(rArr, dbase.rbins, 'bo', ms=10,label='z = 95 nm')

plt.ylabel('Number of photons', fontsize=30)
plt.xlabel('Radius (nm)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
# plt.legend(loc=0, fontsize=20, numpoints=1)
# plt.yscale('log', nonposy='clip')
plt.xlim(0, 400.)
# plt.ylim(5e-3, 0.5)
# plt.ylim(1, 1e4)
plt.show()