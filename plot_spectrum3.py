import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)




dbase.hist(repeat=6250, bins=15000, plotfig=False)

print '=========== Input photon counts ==========='
print 'Energy (eV)\tCounts'
nhard = 0 
print '8495\t' + str(dbase.count_hard_Xray(8495, 1))
nhard += dbase.count_hard_Xray(8495, 1)
print '9628\t' + str(dbase.count_hard_Xray(9628, 1))
nhard += dbase.count_hard_Xray(9628, 1)
print '9713\t' + str(dbase.count_hard_Xray(9713, 1))
nhard += dbase.count_hard_Xray(9713, 1)
print '11442\t' + str(dbase.count_hard_Xray(11442, 1))
nhard += dbase.count_hard_Xray(11442, 1)
print '11587\t' + str(dbase.count_hard_Xray(11587, 1))
nhard += dbase.count_hard_Xray(11587, 1)
print '13422\t' + str(dbase.count_hard_Xray(13422, 1))
nhard += dbase.count_hard_Xray(13422, 1)

print 'Total hard X-rays counts:\t'+str(nhard)
print 'Bremsstrahlung counts\t'+str(dbase.Np_out.sum()-nhard)

ax=plt.subplot(511)
plt.bar(dbase.ebins, dbase.Np_out, color='red', edgecolor='red')
plt.ylabel('Photons/sec', fontsize=20)
plt.title('Input Spectrum: Stage 0', fontsize=25)
# plt.xlabel('Energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=20)
plt.yscale('log', nonposy='clip')
plt.ylim(1e1, 1e5)
plt.grid(True)
# plt.suptitle(r'$Ni (1 \mu{m})$', fontsize=30)
# plt.suptitle(r'$Si (1 \mu{m})$', fontsize=30)
# plt.show()
# 
# 


elesym = 'Al'
dbase.decay_spec(elesym=elesym, t=2e3, plotfig=False)

print '=========== photon counts after Al(2 micro) ==========='
print 'Energy (eV)\tCounts'
nhard = 0 
print '8495\t' + str(dbase.count_hard_Xray(8495, 1))
nhard += dbase.count_hard_Xray(8495, 1)
print '9628\t' + str(dbase.count_hard_Xray(9628, 1))
nhard += dbase.count_hard_Xray(9628, 1)
print '9713\t' + str(dbase.count_hard_Xray(9713, 1))
nhard += dbase.count_hard_Xray(9713, 1)
print '11442\t' + str(dbase.count_hard_Xray(11442, 1))
nhard += dbase.count_hard_Xray(11442, 1)
print '11587\t' + str(dbase.count_hard_Xray(11587, 1))
nhard += dbase.count_hard_Xray(11587, 1)
print '13422\t' + str(dbase.count_hard_Xray(13422, 1))
nhard += dbase.count_hard_Xray(13422, 1)

print 'Total hard X-rays counts:\t'+str(nhard)
print 'Bremsstrahlung counts\t'+str(dbase.Np_out.sum()-nhard)

ax=plt.subplot(512)
plt.bar(dbase.ebins, dbase.Np_out, color='red', edgecolor='red')
plt.ylabel('Photons/sec', fontsize=20)
plt.title('Spectrum after '+r'$Al (2 \mu{m})$'+': Stage 1', fontsize=25)
# plt.xlabel('Energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=20)
plt.yscale('log', nonposy='clip')
plt.ylim(1, 1e5)
plt.grid(True)


rho = 0.7*2.65 + 0.3*8.96
# dbase.decay_spec_CP(chelst=['Si7O14Cu3'], wlst=[1.], denlst=[ rho], t=5e3)
dbase.decay_spec_CP(chelst=['SiO2', 'Cu'], wlst=[0.7, 0.3], denlst=[ 2.65,8.96], t=5e3)

print '=========== photon counts after IC (5 micro, 70% SiO2, 30% Cu) ==========='
print 'Energy (eV)\tCounts'
nhard = 0 
print '8495\t' + str(dbase.count_hard_Xray(8495, 1))
nhard += dbase.count_hard_Xray(8495, 1)
print '9628\t' + str(dbase.count_hard_Xray(9628, 1))
nhard += dbase.count_hard_Xray(9628, 1)
print '9713\t' + str(dbase.count_hard_Xray(9713, 1))
nhard += dbase.count_hard_Xray(9713, 1)
print '11442\t' + str(dbase.count_hard_Xray(11442, 1))
nhard += dbase.count_hard_Xray(11442, 1)
print '11587\t' + str(dbase.count_hard_Xray(11587, 1))
nhard += dbase.count_hard_Xray(11587, 1)
print '13422\t' + str(dbase.count_hard_Xray(13422, 1))
nhard += dbase.count_hard_Xray(13422, 1)

print 'Total hard X-rays counts:\t'+str(nhard)
print 'Bremsstrahlung counts\t'+str(dbase.Np_out.sum()-nhard)


ax=plt.subplot(513)
plt.bar(dbase.ebins, dbase.Np_out, color='red', edgecolor='red')
plt.ylabel('Photons/sec', fontsize=20)
plt.title('Spectrum after '+r'$IC (5 \mu{m})$'+': Stage 2', fontsize=25)
# plt.xlabel('Energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=20)
plt.yscale('log', nonposy='clip')
plt.ylim(1, 1e4)
plt.grid(True)

elesym = 'W'
dbase.decay_spec(elesym=elesym, t=2e3, plotfig=False)

print '=========== photon counts after W (2 micro) ==========='
print 'Energy (eV)\tCounts'
nhard = 0 
print '8495\t' + str(dbase.count_hard_Xray(8495, 1))
nhard += dbase.count_hard_Xray(8495, 1)
print '9628\t' + str(dbase.count_hard_Xray(9628, 1))
nhard += dbase.count_hard_Xray(9628, 1)
print '9713\t' + str(dbase.count_hard_Xray(9713, 1))
nhard += dbase.count_hard_Xray(9713, 1)
print '11442\t' + str(dbase.count_hard_Xray(11442, 1))
nhard += dbase.count_hard_Xray(11442, 1)
print '11587\t' + str(dbase.count_hard_Xray(11587, 1))
nhard += dbase.count_hard_Xray(11587, 1)
print '13422\t' + str(dbase.count_hard_Xray(13422, 1))
nhard += dbase.count_hard_Xray(13422, 1)

print 'Total hard X-rays counts:\t'+str(nhard)
print 'Bremsstrahlung counts\t'+str(dbase.Np_out.sum()-nhard)

ax=plt.subplot(514)
plt.bar(dbase.ebins, dbase.Np_out, color='red', edgecolor='red')
plt.ylabel('Photons/sec', fontsize=20)
plt.title('Spectrum after '+r'$W (2 \mu{m})$'+': Stage 3', fontsize=25)
# plt.xlabel('Energy (eV)', fontsize=30)
ax.tick_params(axis='x', labelsize=0)
ax.tick_params(axis='y', labelsize=20)
plt.yscale('log', nonposy='clip')
plt.ylim(1, 1e4)
plt.grid(True)




Np_in = dbase.Np_out.copy()

elesym = 'Bi'
dbase.decay_spec(elesym=elesym, t=4.1e3, plotfig=False)



print '=========== photon counts after Bi (4.1 micro) ==========='
print 'Energy (eV)\tCounts'
nhard = 0 
print '8495\t' + str(dbase.count_hard_Xray(8495, 1))
nhard += dbase.count_hard_Xray(8495, 1)
print '9628\t' + str(dbase.count_hard_Xray(9628, 1))
nhard += dbase.count_hard_Xray(9628, 1)
print '9713\t' + str(dbase.count_hard_Xray(9713, 1))
nhard += dbase.count_hard_Xray(9713, 1)
print '11442\t' + str(dbase.count_hard_Xray(11442, 1))
nhard += dbase.count_hard_Xray(11442, 1)
print '11587\t' + str(dbase.count_hard_Xray(11587, 1))
nhard += dbase.count_hard_Xray(11587, 1)
print '13422\t' + str(dbase.count_hard_Xray(13422, 1))
nhard += dbase.count_hard_Xray(13422, 1)

print 'Total hard X-rays counts:\t'+str(nhard)
print 'Bremsstrahlung counts\t'+str(dbase.Np_out.sum()-nhard)

ax=plt.subplot(515)
plt.bar(dbase.ebins, (Np_in-dbase.Np_out), color='red', edgecolor='red')
plt.ylabel('Photons/sec', fontsize=20)
plt.title('Spectrum absorbed by the detector after '+r'$Bi (4.1 \mu{m})$'+': Stage 4', fontsize=25)
plt.xlabel('Energy (eV)', fontsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=20)
plt.yscale('log', nonposy='clip')
plt.ylim(1, 1e4)
plt.grid(True)