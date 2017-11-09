import database
import matplotlib.pyplot as plt


pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase1   = database.penelopedbase()
dbase1.read_psf(infname)
N1      = dbase1.energy.size
dbase1.hist(plotfig=False, repeat=6250, bins=15000)


pfx     = 'theta-00'
infname = '/work2/leon/Au_100nm/'+pfx+'/psf-test-short.bin'
dbase2  = database.penelopedbase()
dbase2.read_psf_binary(infname)
N2      = dbase2.energy.size
dbase2.hist(plotfig=False, repeat=int(N1/float(N2)*6250), bins=15000)


# 
ax=plt.subplot()
plt.bar(dbase1.ebins, dbase1.Np, color='red', edgecolor='red', alpha=0.3, lw=5)

plt.bar(dbase2.ebins, dbase2.Np, color='blue', edgecolor='blue', alpha=0.3, lw=5)

plt.ylabel('Photons/sec', fontsize=30)
# plt.title('Spectrum: '+ pfx, fontsize=40)
plt.title('Spectrum', fontsize=40)
plt.xlabel('Energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.yscale('log', nonposy='clip')
plt.ylim(1e0, 1e5)
plt.xlim(0, 15000)
plt.grid(True)
# plt.suptitle(r'$Ni (1 \mu{m})$', fontsize=30)
# plt.suptitle(r'$Si (1 \mu{m})$', fontsize=30)
plt.show()

