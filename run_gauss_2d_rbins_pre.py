import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
dbase.count_r_bins( rmax=800., Nr=401)
# dbase.count_r_bins_cauchy_2d(Nt=22000., gamma=5., rmax=800., Nr=401, plotfig=False)
# dbase.count_r_bins_gauss_2d(Nt=35000., sigma=30., rmax=800., Nr=401, plotfig=False)
# dbase.count_r_bins_exprd_2d(Nt=180., C=68., rmax=800., Nr=401, plotfig=False)
# dbase.count_r_bins_exprd_2d(Nt=200., C=17., rmax=800., Nr=401, plotfig=False)
# dbase.count_r_bins_mix_2d(Nt=22200.,sigma=5., p=0.990, C=67., rmax=800., Nr=401, plotfig=False)
dbase.count_r_bins_mix_2d_3(Nt=19000.,sigma=5., p1=0.983020554066, p2=0.00893655049151, C1=17., C2=68., rmax=800., Nr=401, plotfig=False)


ax=plt.subplot()
plt.plot(dbase.rArr, dbase.rbins, 'ro', ms=10,label='observed')
# plt.plot(dbase.rArr, dbase.rbins_gauss, 'k--',lw=5, ms=10,label='gauss')
dbase.count_r_bins_gauss_2d(Nt=35000., sigma=30, rmax=800., Nr=401, plotfig=False)
plt.plot(dbase.rArr, dbase.rbins_pre, 'k--',lw=5, ms=10,label='Gaussian fit, sigma = 30')
dbase.count_r_bins_gauss_2d(Nt=35000., sigma=60., rmax=800., Nr=401, plotfig=False)
plt.plot(dbase.rArr, dbase.rbins_pre, 'g--',lw=5, ms=10,label='Gaussian fit, sigma = 60')
dbase.count_r_bins_mix_2d_3(Nt=19000.,sigma=5., p1=0.983020554066, p2=0.00893655049151, C1=17., C2=68., rmax=800., Nr=401, plotfig=False)
# dbase.count_r_bins_gauss_2d(Nt=35000., sigma=50., rmax=800., Nr=401, plotfig=False)
plt.plot(dbase.rArr, dbase.rbins_pre, 'b--',lw=5, ms=10,label='Hybrid fit')
# plt.plot(dbase.rArr, dbase.rbins_cauchy, 'k--',lw=5, ms=10,label='gauss')
# plt.plot(dbase.rArr, dbase.rbins_exprd, 'k--',lw=5, ms=10,label='gauss')


plt.ylabel('Number of photons', fontsize=30)
plt.xlabel('Radius (nm)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(loc=0, fontsize=20, numpoints=1)
plt.yscale('log', nonposy='clip')
plt.xlim(0, 400.)
# plt.xlim(0, 100.)
# plt.ylim(1, 0)
plt.ylim(1, 1e4)
plt.show()