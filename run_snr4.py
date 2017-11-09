import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)


dbase.hist(repeat=6250, bins=30000, plotfig=False)

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



rho = 0.7*2.65 + 0.3*8.96
dbase.decay_spec_CP(chelst=['Si7O14Cu3'], wlst=[1.], denlst=[ rho], t=5e3)
# dbase.decay_spec_CP(chelst=['SiO2', 'Cu'], wlst=[0.7, 0.3], denlst=[ 2.65,8.96], t=5e3)

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

# elesym = 'W'
# dbase.decay_spec(elesym=elesym, t=2e3, plotfig=False)
# 
# print '=========== photon counts after W (2 micro) ==========='
# print 'Energy (eV)\tCounts'
# nhard = 0 
# print '8495\t' + str(dbase.count_hard_Xray(8495, 1))
# nhard += dbase.count_hard_Xray(8495, 1)
# print '9628\t' + str(dbase.count_hard_Xray(9628, 1))
# nhard += dbase.count_hard_Xray(9628, 1)
# print '9713\t' + str(dbase.count_hard_Xray(9713, 1))
# nhard += dbase.count_hard_Xray(9713, 1)
# print '11442\t' + str(dbase.count_hard_Xray(11442, 1))
# nhard += dbase.count_hard_Xray(11442, 1)
# print '11587\t' + str(dbase.count_hard_Xray(11587, 1))
# nhard += dbase.count_hard_Xray(11587, 1)
# print '13422\t' + str(dbase.count_hard_Xray(13422, 1))
# nhard += dbase.count_hard_Xray(13422, 1)
# 
# print 'Total hard X-rays counts:\t'+str(nhard)
# print 'Bremsstrahlung counts\t'+str(dbase.Np_out.sum()-nhard)

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
