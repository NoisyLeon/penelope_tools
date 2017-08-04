import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)


dbase.hist(plotfig=True)

elesym = 'Ni'
dbase.decay_spec(elesym=elesym, t=1e3, plotfig=False)
# x = 0.2
# rho = 2.329+3.493*x-0.499*(x**2)
# print rho
# dbase.decay_spec(elesym=elesym, t=1e3, plotfig=False, density=4.8)

ax=plt.subplot(212)
plt.bar(dbase.ebins, dbase.Np_out, color='red', edgecolor='red')
plt.ylabel('Photons/sec', fontsize=30)
plt.title('Attenuated Spectrum', fontsize=40)
plt.xlabel('Energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.yscale('log', nonposy='clip')
plt.ylim(1e4, 1e8)
plt.grid(True)
plt.suptitle(r'$Ni (1 \mu{m})$', fontsize=30)
# plt.suptitle(r'$Si (1 \mu{m})$', fontsize=30)
plt.show()

