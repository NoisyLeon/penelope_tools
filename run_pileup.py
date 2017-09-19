import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.jit(numba.int64[:](numba.float64[:], numba.int64[:], numba.int64, numba.float64))
def _count_unconflicted(t0, assign, Np, tol):
    countArr = np.zeros(Np, dtype=np.int64)
    for ip in xrange(Np):
        tArr = np.ones(t0.size, dtype=np.float64)*(-1.)
        vArr = np.zeros(t0.size, dtype=np.int64)
        for i in xrange(t0.size):
            conflicted = False
            if assign[i] == ip+1:
                for it in xrange(t0.size):
                    if tArr[it]<0.:
                        tArr[it] = t0[i]
                        if not conflicted: vArr[it] = 1
                        break
                    if abs(tArr[it] - t0[i])<tol:
                        conflicted = True
                        vArr[it] = 0
        countArr[ip] = vArr.sum()
    return countArr   


def get_valid_photon(N, tol=0.02, Np=240):
    t0      = np.random.rand(N)
    assign  = np.random.randint(low=1, high=Np, size=N)
    countArr= _count_unconflicted(t0, assign, Np, tol)
    return countArr

tol=0.005
ax=plt.subplot()
for tol in [0.02, 0.01, 0.005]:
    NtArr   = np.arange(30)*1000+1000
    NtunArr = np.zeros(30)
    i=0
    for n in NtArr:
        cArr    = get_valid_photon(N=n, tol=tol)
        Ntun    = cArr.sum()
        NtunArr[i] = Ntun
        i+=1
    
    plt.plot(NtArr, NtunArr, 'o', ms=10, label=str(int(1./tol))+' cps')

inArr = np.loadtxt('tol_50cps.dat')
plt.plot(inArr[:,0], inArr[:,1], 'x-', ms=10, label='50 cps, Mike')
inArr = np.loadtxt('tol_100cps.dat')
plt.plot(inArr[:,0], inArr[:,1], 'x-', ms=10, label='100 cps, Mike')
inArr = np.loadtxt('tol_200cps.dat')
plt.plot(inArr[:,0], inArr[:,1], 'x-', ms=10, label='200 cps, Mike')

plt.ylabel('Number of unconflicted photons', fontsize=30)
plt.xlabel('Number of input photons', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.legend(loc=0, fontsize=20, numpoints=1)
plt.show()
    
    
    
