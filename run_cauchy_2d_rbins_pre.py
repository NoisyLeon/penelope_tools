import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
dbase.count_r_bins( rmax=800., Nr=401)
dbase.count_r_bins_cauchy_2d(Nt=54929.176751690742/2., gamma=5., rmax=800., Nr=401, plotfig=False)



ax=plt.subplot()
plt.plot(dbase.rArr, dbase.rbins, 'bo', ms=10,label='all')
plt.plot(dbase.rArr, dbase.rbins_cauchy, 'k--',lw=5, ms=10,label='cauchy')


plt.ylabel('Number of photons', fontsize=30)
plt.xlabel('Radius (nm)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
# plt.legend(loc=0, fontsize=20, numpoints=1)
# plt.yscale('log', nonposy='clip')
plt.xlim(0, 100.)
# plt.ylim(5e-3, 0.5)
# plt.ylim(1, 1e4)
plt.show()