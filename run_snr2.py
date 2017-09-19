import database
import matplotlib.pyplot as plt
pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)


dbase.hist(repeat=1, bins=30000, plotfig=False)

elesym = 'W'
dbase.decay_spec(elesym=elesym, t=1e3, plotfig=False)
cbins = dbase.ebins
dbins=dbase.ebins[1] - dbase.ebins[0]
energy = dbase.energy

# 8495	687500
# 9628	1112500
# 9713	9225000
# 11442	3675000
# 11587	1737500
# 13422	700000

n = dbase.Np
nout=dbase.Np_out

nhard=0
ind = (cbins > (8495.-dbins))*( cbins<(8495.+dbins) )
imax = n[ind].argmax()
e1 =(cbins[ind])[imax]
nhard+=nout[ind].max()*6250
print str(e1)+'\t' +str(n[ind].max()*6250)+'\t'+str(nout[ind].max()*6250)

#
ind = (cbins > (9628-dbins))*( cbins<(9628+dbins) )
imax = n[ind].argmax()
e1 = (cbins[ind])[imax]
nhard+=nout[ind].max()*6250
print str(e1)+'\t' +str(n[ind].max()*6250)+'\t'+str(nout[ind].max()*6250)

ind = (cbins > (9713-dbins))*( cbins<(9713+dbins) )
imax = n[ind].argmax()
e1 = (cbins[ind])[imax]
nhard+=nout[ind].max()*6250
print str(e1)+'\t' +str(n[ind].max()*6250)+'\t'+str(nout[ind].max()*6250)

ind = (cbins > (11443-dbins))*( cbins<(11443+dbins) )
imax = n[ind].argmax()
e1 = (cbins[ind])[imax]
nhard+=nout[ind].max()*6250
print str(e1)+'\t' +str(n[ind].max()*6250)+'\t'+str(nout[ind].max()*6250)

ind = (cbins > (11587-dbins))*( cbins<(11587+dbins) )
imax = n[ind].argmax()
e1 = (cbins[ind])[imax]
nhard+=nout[ind].max()*6250
print str(e1)+'\t' +str(n[ind].max()*6250)+'\t'+str(nout[ind].max()*6250)

ind = (cbins > (13423-dbins))*( cbins<(13423+dbins) )
imax = n[ind].argmax()
e1 = (cbins[ind])[imax]
nhard+=nout[ind].max()*6250
print str(e1)+'\t' +str(n[ind].max()*6250)+'\t'+str(nout[ind].max()*6250)
# 
#

print 'Bremsstrahlung\t', nout[cbins>5000.].sum()*6250-nhard

# 
# 
# ax=plt.subplot(212)
# plt.bar(dbase.ebins, dbase.Np_out, color='red', edgecolor='red')
# plt.ylabel('Photons/sec', fontsize=30)
# plt.title('Attenuated Spectrum', fontsize=40)
# plt.xlabel('Energy (eV)', fontsize=30)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# plt.yscale('log', nonposy='clip')
# plt.ylim(1e4, 1e8)
# plt.grid(True)
# plt.suptitle(r'$Ni (1 \mu{m})$', fontsize=30)
# # plt.suptitle(r'$Si (1 \mu{m})$', fontsize=30)
# plt.show()