
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

pfx = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
energy = inArr[:, 1]
# energy = np.repeat(energy, 6250)

n, bins, patches = plt.hist(energy, bins=15000, normed=False, facecolor='blue', edgecolor='blue')
e5000  = energy[energy>5000.]

dbins=bins[1] - bins[0]
cbins = (bins[1:] + bins[:-1])/2
dbins = 10*dbins
# print n[(cbins > (8495.-dbins))*( cbins<(8495.+dbins) )], cbins[(cbins > (8495.-dbins))*( cbins<(8495.+dbins) )]
# print n[(cbins > (9628-dbins))*( cbins<(9628+dbins) )], cbins[(cbins > (9628-dbins))*( cbins<(9628+dbins) )]
# print n[(cbins > (9711-dbins))*( cbins<(9711+dbins) )], cbins[(cbins > (9711-dbins))*( cbins<(9711+dbins) )]
# print n[(cbins > (11443-dbins))*( cbins<(11443+dbins) )], cbins[(cbins > (11443-dbins))*( cbins<(11443+dbins) )]
# print n[(cbins > (11587-dbins))*( cbins<(11587+dbins) )], cbins[(cbins > (11587-dbins))*( cbins<(11587+dbins) )]
# print n[(cbins > (13423-dbins))*( cbins<(13423+dbins) )], cbins[(cbins > (13423-dbins))*( cbins<(13423+dbins) )]

nhard=0
imax = n[(cbins > (8495.-dbins))*( cbins<(8495.+dbins) )].argmax()
e1 =(cbins[(cbins > (8495.-dbins))*( cbins<(8495.+dbins) )])[imax]
nhard+=energy[(energy>e1-1)*(energy<e1+1)].size*6250
print str(e1)+'\t' +str(energy[(energy>e1-1)*(energy<e1+1)].size*6250)


imax = n[(cbins > (9628-dbins))*( cbins<(9628+dbins) )].argmax()
e1 = (cbins[(cbins > (9628-dbins))*( cbins<(9628+dbins) )])[imax]
nhard+=energy[(energy>e1-1)*(energy<e1+1)].size*6250
print str(e1)+'\t' +str(energy[(energy>e1-1)*(energy<e1+1)].size*6250)

imax = n[(cbins > (9711-dbins))*( cbins<(9711+dbins) )].argmax()
e1 = (cbins[(cbins > (9711-dbins))*( cbins<(9711+dbins) )])[imax]
nhard+=energy[(energy>e1-1)*(energy<e1+1)].size*6250
print str(e1)+'\t' +str(energy[(energy>e1-1)*(energy<e1+1)].size*6250)

imax = n[(cbins > (11443-dbins))*( cbins<(11443+dbins) )].argmax()
e1 = (cbins[(cbins > (11443-dbins))*( cbins<(11443+dbins) )])[imax]
nhard+=energy[(energy>e1-1)*(energy<e1+1)].size*6250
print str(e1)+'\t' +str(energy[(energy>e1-1)*(energy<e1+1)].size*6250)

imax = n[(cbins > (11587-dbins))*( cbins<(11587+dbins) )].argmax()
e1 = (cbins[(cbins > (11587-dbins))*( cbins<(11587+dbins) )])[imax]
nhard+=energy[(energy>e1-1)*(energy<e1+1)].size*6250
print str(e1)+'\t' +str(energy[(energy>e1-1)*(energy<e1+1)].size*6250)

imax = n[(cbins > (13423-dbins))*( cbins<(13423+dbins) )].argmax()
e1 = (cbins[(cbins > (13423-dbins))*( cbins<(13423+dbins) )])[imax]
nhard+=energy[(energy>e1-1)*(energy<e1+1)].size*6250
print str(e1)+'\t' +str(energy[(energy>e1-1)*(energy<e1+1)].size*6250)


print 'Bremsstrahlung\t', e5000.size*6250-nhard
