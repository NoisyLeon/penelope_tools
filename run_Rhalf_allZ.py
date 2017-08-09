import database
import matplotlib.pyplot as plt
import numpy as np
plst = ['I_1nA_E_30keV','I_5nA_E_30keV',  'I_10nA_E_30keV', 'I_100nA_E_30keV']
ax=plt.subplot()
clst=['k', 'b', 'r', 'g']
rArr    = np.array([])
for pfx in plst:

    infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
    dbase   = database.penelopedbase()
    dbase.read_psf(infname)
    dbase.get_char_radius(zmin=None, ratio=0.9)
    rArr    = np.append(rArr, dbase.cr)
    

IArr = np.array([1., 5., 10., 100.])

plt.plot(IArr, rArr, 'o', lw=3, ms=8, label='50 %')
    # i+=1
    
    
plt.legend(loc=0, fontsize=20)
plt.ylabel('Characteristic Diameter (nm)', fontsize=30)
plt.xlabel('Current (nA)', fontsize=30)
# plt.title('90%)', fontsize=40)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.xlim(0., 100.)
plt.show()
# 
# 
# alpha=np.e
# i=0
# for pfx in plst:
#     plt.plot(zArr, gammaLst[i]*np.sqrt(alpha-1), clst[i]+'o--', lw=3, ms=8, label=pfx)
#     i+=1
#     
# plt.legend(loc=0, fontsize=20)
# plt.ylabel('Re (nm)', fontsize=30)
# plt.xlabel('Z (nm)', fontsize=30)
# # plt.title('2 sigma (95%)', fontsize=40)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# plt.xlim(0., 100.)
# plt.show()
# 
# alpha=2.
# i=0
# for pfx in plst:
#     plt.plot(zArr, gammaLst[i]*np.sqrt(alpha-1), clst[i]+'o--', lw=3, ms=8, label=pfx)
#     i+=1
#     
# plt.legend(loc=0, fontsize=20)
# plt.ylabel('Rhalf (nm)', fontsize=30)
# plt.xlabel('Z (nm)', fontsize=30)
# # plt.title('2 sigma (95%)', fontsize=40)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# plt.xlim(0., 100.)
# plt.show()