import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
dbase.count_r_bins( rmax=800., Nr=401)

dbase2   = database.penelopedbase()
dbase2.read_r_bins_mat(rmax=800., Nr=401, infname='/home/leon/code/InverseTransformSampling/xray_rbins.mat')
# dbase2.read_mat('/home/leon/code/InverseTransformSampling/xray1.mat')
# dbase2.read_mat('/home/leon/code/InverseTransformSampling/xray2.mat')
# dbase2.read_mat('/home/leon/code/InverseTransformSampling/xray3.mat')
# dbase2.read_mat('/home/leon/code/InverseTransformSampling/xray4.mat')
# 
# dbase2.count_r_bins( rmax=800., Nr=401)



# dbase.count_r_bins_mix_2d(Nt=22200.,sigma=5., p=0.990, C=67., rmax=800., Nr=401, plotfig=False)


ratio = dbase.rbins[0]/dbase2.rbins[0]
ratio = 1.
ax=plt.subplot()
plt.plot(dbase.rArr, dbase.rbins, 'ro', ms=10,label='observed')
plt.plot(dbase2.rArr, dbase2.rbins*ratio, 'bo', ms=10,label='predicted')
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