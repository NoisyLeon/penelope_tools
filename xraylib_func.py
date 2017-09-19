
import xraylib
import xraylib_np
import numpy as np
import matplotlib.pyplot as plt

def get_mu(energy, elesym=None, elenumb=None):
    if elesym == None and elenumb ==None:
        raise ValueError('Element symbol or number need to be specified!')
    if elesym!=None: elenumb = xraylib.SymbolToAtomicNumber(elesym)
    mu = xraylib.CS_Total(elenumb, energy) * xraylib.ElementDensity(elenumb)
    return mu

def get_mu_np(energy, elesym=None, elenumb=None):
    """
    Compute attenuation coefficients for a energy array given an element
    ======================================================================
    Parameters:
    energy  - input energy array (type of np.ndarray, unit: keV)
    elesym  - element symbol
    elenumb - element's atomic number
    ======================================================================
    """
    if elesym == None and elenumb ==None:
        raise ValueError('Element symbol or number need to be specified!')
    if elesym!=None: elenumb = xraylib.SymbolToAtomicNumber(elesym)
    mu = xraylib_np.CS_Total(np.array([elenumb]), energy) * xraylib.ElementDensity(elenumb)
    return mu

def get_muoverrho_np(energy, elesym=None, elenumb=None):
    """
    Compute attenuation coefficients for a energy array given an element
    ======================================================================
    Parameters:
    energy  - input energy array (type of np.ndarray, unit: keV)
    elesym  - element symbol
    elenumb - element's atomic number
    ======================================================================
    """
    if elesym == None and elenumb ==None:
        raise ValueError('Element symbol or number need to be specified!')
    if elesym!=None: elenumb = xraylib.SymbolToAtomicNumber(elesym)
    muoverrho = xraylib_np.CS_Total(np.array([elenumb]), energy) 
    return muoverrho

def get_mu_np_CP(energy, chemicalform, density):
    muArr = np.array([])
    for e in energy:
        mu = xraylib.CS_Total_CP(chemicalform, e) *density
        muArr=np.append(muArr, mu)
    return muArr

def get_muoverrho_np_CP(energy, chemicalform):
    muorhoArr = np.array([])
    for e in energy:
        muoverrho = xraylib.CS_Total_CP(chemicalform, e) 
        muorhoArr=np.append(muorhoArr, muoverrho)
    return muorhoArr




# 
# print 'Hf', get_mu(elesym='Hf', energy=1.)
# print 'Ta', get_mu(elesym='Ta', energy=1.)
# print 'W', get_mu(elesym='W', energy=1.)
# print 'Bi', get_mu(elesym='Bi', energy=1.)
# print 'Au', get_mu(elesym='Au', energy=1.)
# print 'Pb', get_mu(elesym='Pb', energy=1.)
# print 'Zn', get_mu(elesym='Zn', energy=1.)
# print 'Cu', get_mu(elesym='Cu', energy=1.)
# print 'Pt', get_mu(elesym='Cu', energy=1.)
# print 'Cu', get_mu(elesym='Cu', energy=1.)
# 
# t   = 1.e5
# t   = t/1.e7
# eArr= np.arange(30000.)/1000. + .001
# mu  = (get_mu_np(energy=eArr, elesym='Al'))[0,:]
# 
# ratio= np.exp(-mu*t)
# 
# plt.plot(eArr*1000., ratio, '-', lw=3)
# plt.show()


        