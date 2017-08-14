import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
dbase.count_r_bins( rmax=800., Nr=401)
# dbase.count_r_bins_cauchy_2d(Nt=18000., gamma=5., rmax=800., Nr=401, plotfig=False)
# dbase.count_r_bins_gauss_2d(Nt=12400., sigma=4., rmax=800., Nr=401, plotfig=False)
# dbase.count_r_bins_exprd_2d(Nt=170., C=66., rmax=800., Nr=401, plotfig=False)
dbase.count_r_bins_mix_2d(Nt=18170.,sigma=4., p=0.9906439185470556, C=68.9902980492, rmax=800., Nr=401, plotfig=False)



ax=plt.subplot()
plt.plot(dbase.rArr, dbase.rbins, 'bo', ms=10,label='all')
# plt.plot(dbase.rArr, dbase.rbins_gauss, 'k--',lw=5, ms=10,label='gauss')
plt.plot(dbase.rArr, dbase.rbins_mix, 'k--',lw=5, ms=10,label='gauss')
# plt.plot(dbase.rArr, dbase.rbins_cauchy, 'k--',lw=5, ms=10,label='gauss')
# plt.plot(dbase.rArr, dbase.rbins_exprd, 'k--',lw=5, ms=10,label='gauss')


plt.ylabel('Number of photons', fontsize=30)
plt.xlabel('Radius (nm)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
# plt.legend(loc=0, fontsize=20, numpoints=1)
plt.yscale('log', nonposy='clip')
plt.xlim(0, 400.)
# plt.xlim(0, 100.)
# plt.ylim(5e-3, 0.5)
# plt.ylim(1, 1e4)
plt.show()