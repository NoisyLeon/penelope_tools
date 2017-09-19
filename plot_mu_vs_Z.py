import database
import matplotlib.pyplot as plt
import numpy as np
import xraylib_func
import xraylib
muArr = np.zeros(98)

for i in np.arange(98)+1:
    print i
    muArr[i-1]=xraylib_func.get_mu(energy=9.713, elenumb=i)
ax=plt.subplot()
plt.yscale('log', nonposy='clip')
plt.plot(np.arange(98)+1, muArr, 'o', ms=8)


elemlst = ['Hf', 'Ta', 'W', 'Bi', 'Au', 'Pb', 'Zn', 'Cu', 'Pt', 'Si', 'Al']
i=0
for elem in elemlst:
    mu  = xraylib_func.get_mu(energy=9.713, elesym=elem)
    elenumb = xraylib.SymbolToAtomicNumber(elem)
    i+=1
    if i < 5:
        plt.plot(elenumb, mu, 'o', ms=15, label=elem)
    else:
        plt.plot(elenumb, mu, '^', ms=15, label=elem)

plt.ylabel(r'$\mu (cm^{-1})$', fontsize=30)
plt.xlabel('Atomic number (Z)', fontsize=30)
plt.title('Energy = 9.713 keV', fontsize=35)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(numpoints=1, fontsize=15, loc=0)
# ax.grid(color='k', linestyle='--', linewidth=1, axis='x')

major_ticks = np.arange(10)*10+10                            
minor_ticks = np.arange(50)*2+2                                            

ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)                                           
# ax.set_yticks(major_ticks)                                                       
# ax.set_yticks(minor_ticks, minor=True)                                           

# and a corresponding grid                                                       

# ax.grid(which='x')                                                            

# or if you want differnet settings for the grids:                               
ax.grid(which='minor', alpha=0.5,lw=1, axis='x')                                           
ax.grid(which='major', alpha=1, lw=1, ls='-', axis='x')

plt.show()