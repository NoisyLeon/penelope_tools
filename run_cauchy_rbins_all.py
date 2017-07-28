import database
import matplotlib.pyplot as plt
import numpy as np
plst = ['I_1nA_E_30keV','I_5nA_E_30keV',  'I_10nA_E_30keV', 'I_100nA_E_30keV']
# plst = ['I_100nA_E_30keV']

clst=['k', 'b', 'r', 'g']

gammaLst=[]
gmaxLst=[]
gminLst=[]
for pfx in plst:
    gammaArr    = np.array([]); gmaxArr    = np.array([]); gminArr    = np.array([])
    zArr    = np.arange(19)*5.
    for zmin in zArr:
        print zmin, pfx
        infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
        dbase   = database.penelopedbase()
        dbase.read_psf(infname)
        dbase.count_r_bins(zmin=zmin, rmax=800., Nr=401)
        
        dbase.cauchy_rbins_fit(plotrmax=100., normtype=2)
        gammaArr    = np.append(gammaArr, dbase.gamma_cauchy_rbins)
        gmaxArr     = np.append(gmaxArr, dbase.gmax)
        gminArr     = np.append(gminArr, dbase.gmin)
    gammaLst.append(gammaArr)
    gmaxLst.append(gmaxArr)
    gminLst.append(gminArr)

zArr = np.arange(19)*5. +5.
i=0
ax=plt.subplot()
for pfx in plst:
    # plt.plot(zArr, gammaLst[i], clst[i]+'o--', lw=3, ms=8, label=pfx)
    plt.errorbar(zArr, gammaLst[i], fmt=clst[i]+'o--', yerr=[gammaLst[i]-gminLst[i], gmaxLst[i]-gammaLst[i]], lw=3, ms=8, label=pfx)
    i+=1
    
plt.legend(loc=0, fontsize=20)
plt.ylabel('gamma (nm)', fontsize=30)
plt.xlabel('Z (nm)', fontsize=30)
# plt.title('2 sigma (95%)', fontsize=40)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
plt.xlim(0., 100.)
plt.show()


i=0
ax=plt.subplot()
for pfx in plst:
    plt.plot(zArr, gammaLst[i], clst[i]+'o--', lw=3, ms=8, label=pfx)
    # plt.errorbar(zArr, gammaLst[i], fmt=clst[i]+'o--', yerr=[gammaLst[i]-gminLst[i], gmaxLst[i]-gammaLst[i]], lw=3, ms=8, label=pfx)
    i+=1
    
plt.legend(loc=0, fontsize=20)
plt.ylabel('gamma (nm)', fontsize=30)
plt.xlabel('Z (nm)', fontsize=30)
# plt.title('2 sigma (95%)', fontsize=40)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
plt.xlim(0., 100.)
plt.show()


alpha=np.e
i=0
ax=plt.subplot()
for pfx in plst:
    plt.plot(zArr, 2.*gammaLst[i]*np.sqrt(alpha-1), clst[i]+'o--', lw=3, ms=8, label=pfx)
    i+=1
    
plt.legend(loc=0, fontsize=20)
plt.ylabel('1/e Diameter(nm)', fontsize=30)
plt.xlabel('Z (nm)', fontsize=30)
# plt.title('2 sigma (95%)', fontsize=40)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
plt.xlim(0., 100.)
plt.show()

alpha=2.
i=0
ax=plt.subplot()
for pfx in plst:
    plt.plot(zArr, gammaLst[i]*np.sqrt(alpha-1), clst[i]+'o--', lw=3, ms=8, label=pfx)
    i+=1
    
plt.legend(loc=0, fontsize=20)
plt.ylabel('Rhalf (nm)', fontsize=30)
plt.xlabel('Z (nm)', fontsize=30)
# plt.title('2 sigma (95%)', fontsize=40)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
plt.xlim(0., 100.)
plt.show()