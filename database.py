import numpy as np
import asdf
import subprocess, os
import copy
import matplotlib.pyplot as plt
import shutil


@numba.jit(int32[:,:](int32[:,:], float64[:], float64[:], float64[:], float64[:], float64) )
def _count_2D_points(nArr, xg, yg, xin, yin, dx):
    for ix in xrange(xg.size):
        for iy in xrange(yg.size):
            x   = xg[ix]; y = yg[iy]
            xmin = x - dx; xmax = x + dx
            ymin = y - dx; ymax = y + dx
            N=np.where((xin>=xmin)*(xin<xmax)*(yin>=ymin)*(yin<ymax))[0].size
            nArr[ix, iy] = N
    return nArr


def _cauchy_2d_fit(xgg, ygg, nArr, N0Arr, gammaArr):
    rms = 999
    gamma_min = 0.
    N_min   = 0.
    for N in N0Arr:
        # N   = iN*100 + 26000
        for gamma in gammaArr:
            # gamma = ig*0.5 + 45.
            pdf = 1./np.pi/2.*(gamma/(((xgg)**2+(ygg)**2+gamma**2)**1.5) )
            nArr_pre = pdf*N
            temp = (nArr_pre - nArr)**2
            temp = temp[nArr>0]
            rms_temp = np.sqrt(np.mean(temp))
            if rms_temp < rms:
                rms = rms_temp; gamma_min = gamma; Nmin = N
        print N, gamma_min, rms
    return Nmin, gamma_min, rms

class penelopedbase(object):
    """
    An object to handle output from PENELOPE
    ===========================================================================================================
    Parameters:
    IE      - intial energy (default: 30 keV)
    I       - current (default: 10 nA)
    h       - thickness of the foil (default: 100 nm)
    symbol  - chemical symbol of the foil (default: Au)
    x, y, z - position of particle (unit: nm)
    u, v, w - particle direction in terms of unit vector
    energy  - particle energy (unit: eV)
    ===========================================================================================================
    """
    def __init__(self, IE=30, I=10, h=100, symbol='Au', x = np.array([]), y=np.array([]), z=np.array([]),
            u=np.array([]), v=np.array([]), w=np.array([]), energy=np.array([])):
        
        
        self.IE     = IE
        self.I      = I
        self.h      = h
        self.symbol = symbol
        ###
        # phase-space data
        ###
        self.x      = x
        self.y      = y
        self.z      = z
        self.energy = energy
        self.u      = u
        self.v      = v
        self.w      = w
        return
    
    def read_psf(self, infname, factor=1e7):
        """
        Read phase-space file
        """
        inArr = np.loadtxt(infname)
        self.x = inArr[:, 2]*factor; self.y = inArr[:, 3]*factor; self.z = inArr[:, 4]*factor
        self.u = inArr[:, 5]; self.v = inArr[:, 6]; self.w = inArr[:, 7]
        self.energy = inArr[:, 1]
        return
    
    def cauchy_2d_fit(self, zmin, zmax=None, dN=100, dgamma=0.5):
        if zmax == None: zmax = zmin + 10.
        
        
    
    # 
    # def load(self, infname):
    #     """
    #     Load ASDF file
    #     """
    #     self.tree.update((asdf.AsdfFile.open(infname)).tree)