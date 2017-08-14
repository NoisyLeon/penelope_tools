import numpy as np
import numba
import asdf
# import subprocess, os
# import copy
import matplotlib.pyplot as plt
# import shutil
import xraylib_func


@numba.jit(numba.int32[:,:](numba.int32[:,:], numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:], numba.float64) )
def _count_2D_points(nArr, xg, yg, xin, yin, dx):
    for ix in xrange(xg.size):
        for iy in xrange(yg.size):
            x   = xg[ix]; y = yg[iy]
            xmin = x - dx; xmax = x + dx
            ymin = y - dx; ymax = y + dx
            N=np.where((xin>=xmin)*(xin<xmax)*(yin>=ymin)*(yin<ymax))[0].size
            nArr[ix, iy] = N
    return nArr


def _cauchy_2d_fit(xgg, ygg, nArr, N0Arr, gammaArr, normtype=1):
    rms         = 999
    gamma_min   = 0.
    N_min       = 0.
    for N in N0Arr:
        # N   = iN*100 + 26000
        for gamma in gammaArr:
            # gamma = ig*0.5 + 45.
            print N, gamma
            pdf         = 1./np.pi/2.*(gamma/(((xgg)**2+(ygg)**2+gamma**2)**1.5) )
            nArr_pre    = pdf*N
            if normtype == 2:
                temp        = (nArr_pre - nArr)**2
                temp        = temp[nArr>0]
                rms_temp    = np.sqrt(np.mean(temp))
            elif normtype == 1:
                temp    = np.abs(nArr_pre - nArr)
                temp    = temp[nArr>0]
                rms_temp= np.mean(temp)
            else:
                temp    = (np.abs(nArr_pre - nArr))**normtype
                temp    = temp[nArr>0]
                rms_temp= (np.mean(temp))**(1./normtype)
            if rms_temp < rms:
                rms = rms_temp; gamma_min = gamma; Nmin = N
        # print N, gamma_min, rms
    return Nmin, gamma_min, rms

def _gauss_2d_fit(xgg, ygg, nArr, N0Arr, sigmaArr, normtype=1):
    rms         = 999
    sigma_min   = 0.
    N_min       = 0.
    for N in N0Arr:
        # N   = iN*100 + 26000
        for sigma in sigmaArr:
            # gamma = ig*0.5 + 45.
            print N, sigma
            pdf         = np.exp( -0.5*((xgg/sigma)**2) - 0.5*((ygg/sigma)**2) ) / 2./np.pi/(sigma**2)
            nArr_pre    = pdf*N
            if normtype == 2:
                temp        = (nArr_pre - nArr)**2
                temp        = temp[nArr>0]
                rms_temp    = np.sqrt(np.mean(temp))
            elif normtype == 1:
                temp    = np.abs(nArr_pre - nArr)
                temp    = temp[nArr>0]
                rms_temp= np.mean(temp)
            else:
                temp    = (np.abs(nArr_pre - nArr))**normtype
                temp    = temp[nArr>0]
                rms_temp= (np.mean(temp))**(1./normtype)
            if rms_temp < rms:
                rms = rms_temp; sigma_min = sigma; Nmin = N
    return Nmin, sigma_min, rms

def _exprd_2d_fit(xgg, ygg, nArr, N0Arr, CArr, r=75., normtype=1):
    rms         = 999
    sigma_min   = 0.
    N_min       = 0.
    R           = np.sqrt(xgg**2+ygg**2)
    ind         = R > 75.
    for N in N0Arr:
        for C in CArr:
            pdf         = 1./2./np.pi/C*np.exp( -R/C)
            nArr_pre    = pdf*N
            
            if normtype == 2:
                temp        = (nArr_pre - nArr)**2
                temp        = temp[(nArr>0)*(ind)]
                rms_temp    = np.sqrt(np.mean(temp))
            elif normtype == 1:
                temp    = np.abs(nArr_pre - nArr)
                temp    = temp[(nArr>0)*(ind)]
                rms_temp= np.mean(temp)
            else:
                temp    = (np.abs(nArr_pre - nArr))**normtype
                temp    = temp[(nArr>0)*(ind)]
                rms_temp= (np.mean(temp))**(1./normtype)
            if rms_temp < rms:
                rms = rms_temp; Cmin = C; Nmin = N
            print N, C, rms_temp
    return Nmin, Cmin, rms

def _mix_2d_fit(xgg, ygg, nArr, N0Arr, sigmaArr, pArr, CArr, normtype=1):
    sigma   = 4.0
    rms         = 999
    sigma_min   = 0.
    N_min       = 0.
    for N in N0Arr:
        for sigma in sigmaArr:
            for p in pArr:
                for C in CArr:
                    pdf1    = np.exp( -0.5*((xgg/sigma)**2) - 0.5*((ygg/sigma)**2) ) / 2./np.pi/(sigma**2)
                    pdf2    = 1./4/(C**2)*np.exp( -np.abs(xgg)/C - np.abs(ygg)/C)
                    pdf     = pdf1*(1-p) + pdf2*p
                    nArr_pre    = pdf*N
                    if normtype == 2:
                        temp        = (nArr_pre - nArr)**2
                        temp        = temp[nArr>0]
                        rms_temp    = np.sqrt(np.mean(temp))
                    elif normtype == 1:
                        temp    = np.abs(nArr_pre - nArr)
                        temp    = temp[nArr>0]
                        rms_temp= np.mean(temp)
                    else:
                        temp    = (np.abs(nArr_pre - nArr))**normtype
                        temp    = temp[nArr>0]
                        rms_temp= (np.mean(temp))**(1./normtype)
                    if rms_temp < rms:
                        rms = rms_temp; sigma_min = sigma; Nmin = N; pmin = p; Cmin = C
        
    return pmin, Cmin, Nmin, sigma_min, rms
    


def _cauchy_2d_fit_fix_N0(xgg, ygg, nArr, gammaArr, N0, normtype=1):
    rms         = 999
    gamma_min   = 0.
    R           = np.sqrt((xgg)**2+(ygg)**2)
    for gamma in gammaArr:
        print gamma, rms
        pdf         = 1./np.pi/2.*(gamma/(((xgg)**2+(ygg)**2+gamma**2)**1.5) )
        Nt          = N0*2.*np.pi*(gamma**2)
        nArr_pre    = pdf*Nt
        if normtype == 2:
            temp        = (nArr_pre - nArr)**2
            # temp        = temp[nArr>0]
            temp        = temp[(nArr>0)*(R>25)*(R<400.)]
            rms_temp    = np.sqrt(np.mean(temp))
        elif normtype == 1:
            temp    = np.abs(nArr_pre - nArr)
            # temp    = temp[nArr>0]
            temp    = temp[(nArr>0)*(R>25)*(R<400.)]
            rms_temp= np.mean(temp)
        else:
            temp    = (np.abs(nArr_pre - nArr))**normtype
            # temp    = temp[nArr>0]
            temp    = temp[(nArr>0)*(R>25)*(R<400.)]
            rms_temp= (np.mean(temp))**(1./normtype)
        if rms_temp < rms:
            rms = rms_temp; gamma_min = gamma
    return gamma_min, rms

def _cauchy_rbins_fit(rg, rbins, A0, gammaArr, normtype=1):
    rms         = 1e9
    gamma_min   = 0.
    Amin        = 0.
    rmsArr      = np.array([])
    for gamma in gammaArr:
        Aarr = A0*np.pi*gamma/2. + np.arange(100.)*A0*np.pi*gamma/100.
        rms_g= 1e9
        for A in Aarr:
            pdf     = A/np.pi*(gamma/((rg)**2+gamma**2) )
            if normtype == 2:
                temp    = (pdf - rbins)**2
                temp    = temp[rbins>0]
                # temp2   = rbins[rbins>0]
                # rms_temp= np.sqrt(np.mean(temp/(temp2**2)))
                rms_temp= np.sqrt(np.mean(temp))
            elif normtype == 1:
                temp    = np.abs(pdf - rbins)
                temp    = temp[rbins>0]
                rms_temp= np.mean(temp)
            else:
                temp    = (np.abs(pdf - rbins))**normtype
                temp    = temp[rbins>0]
                rms_temp= (np.mean(temp))**(1./normtype)
            # else:
            #     raise ValueError('Not supported norm type: '+str(normtype))
            if rms_temp < rms:
                rms = rms_temp; gamma_min = gamma; Amin = A
            if rms_temp < rms_g:
                rms_g = rms_temp
        rmsArr= np.append(rmsArr, rms_g)
    return Amin, gamma_min, rms, rmsArr

def _gauss_rbins_fit(rg, rbins, sigmaArr, normtype=1):
    rms         = 1e9
    sigma_min   = 0.
    Amin        = 1.
    rmsArr      = np.array([])
    
    for sigma in sigmaArr:
        Aarr = np.arange(100.)*0.05+.5
        for A in Aarr:
            pdf     = A/np.sqrt(2*np.pi)/sigma*np.exp(-(rg)**2/2./(sigma**2))
            if normtype == 2:
                temp    = (pdf - rbins)**2
                temp    = temp[rbins>0]
                # temp2   = rbins[rbins>0]
                # rms_temp= np.sqrt(np.mean(temp/(temp2**2)))
                rms_temp= np.sqrt(np.mean(temp))
            elif normtype == 1:
                temp    = np.abs(pdf - rbins)
                temp    = temp[rbins>0]
                rms_temp= np.mean(temp)
            else:
                temp    = (np.abs(pdf - rbins))**normtype
                temp    = temp[rbins>0]
                rms_temp= (np.mean(temp))**(1./normtype)
            if rms_temp < rms:
                rms = rms_temp; sigma_min = sigma; Amin=A
        # rmsArr= np.append(rmsArr, rms_g)
    # print sigma_min, rms, Amin
    return sigma_min, rms, Amin
    

def _cauchy_2d(xg, yg, gamma, mux, muy):
    xgg, ygg = np.meshgrid(xg, yg, indexing='ij')
    pdf = 1./np.pi/2.*(gamma/(((xgg-mux)**2+(ygg-muy)**2+gamma**2)**1.5) )
    return pdf

def _gauss_2d(xg, yg, sigma, mux, muy):
    xgg, ygg    = np.meshgrid(xg, yg, indexing='ij')
    pdf         = np.exp( -0.5*(((xgg-mux)/sigma)**2) - 0.5*(((ygg-muy)/sigma)**2) ) / 2./np.pi/(sigma**2)
    return pdf


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
    def __init__(self, xmin=-800., xmax=800., Nx=1601, ymin=-800., ymax=800., Ny=1601, zmin=-5., zmax=95., Nz=10, IE=30, I=10, h=100,
            symbol='Au', x = np.array([]), y=np.array([]), z=np.array([]), u=np.array([]), v=np.array([]), w=np.array([]), energy=np.array([])):
        self.IE     = IE
        self.I      = I
        self.h      = h
        self.symbol = symbol
        ###
        # grid points
        ###
        self.Nx     = Nx
        self.Ny     = Ny
        self.Nz     = Nz
        self.xgrid  = np.mgrid[xmin:xmax:Nx*1j]
        self.ygrid  = np.mgrid[ymin:ymax:Ny*1j]
        self.zgrid  = np.mgrid[zmin:zmax:Nz*1j]
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
    
    def save_2d_points(self, outfname):
        np.save(outfname, self.nArr)
        return
    
    def load_2d_points(self, infname):
        self.nArr = np.load(infname)
        return
    
    def count_2d_points(self, zmin=None, zmax=None, outfname=None):
        """
        Count data points for predefined grid
        """
        if zmin != None:
            if zmax == None: zmax = zmin + 10.
            ind     = (self.z >= zmin)*(self.z <= zmax)
            xin     = self.x[ind];  yin     = self.y[ind]; zin      = self.z[ind]
            self.z2d= (zmin+zmax)/2.
        else:
            xin     = self.x.copy();  yin     = self.y.copy(); zin      = self.z.copy()
        dx          = (self.xgrid[1] - self.xgrid[0])/2.
        nArr        = np.zeros((self.xgrid.size, self.ygrid.size), np.int32)
        try:
            self.z2d
            print 'Counting 2D points for z =', self.z2d,' nm'
        except:
            print 'Counting all points as 2D'
        self.nArr   = _count_2D_points(nArr, self.xgrid, self.ygrid, xin, yin, dx)
        print 'End Counting 2D points.'
        if outfname != None: self.save_2d_points(outfname=outfname)
        return
    
    def count_r_bins(self, rmax, Nr, zmin=None, rmin=0., zmax=None, plotfig=False):
        """
        Count data points for radius bins
        """
        rArr    = np.mgrid[rmin:rmax:Nr*1j]
        if zmin != None:
            if zmax == None: zmax = zmin + 10.
            ind     = (self.z >= zmin)*(self.z <= zmax)
            xin     = self.x[ind];  yin     = self.y[ind]; zin      = self.z[ind]
        else:
            xin     = self.x.copy();  yin     = self.y.copy(); zin      = self.z.copy()
        R       = np.sqrt(xin**2+yin**2)
        self.rbins   = np.zeros(rArr.size-1)
        self.area   = np.zeros(rArr.size-1)
        for ir in xrange(Nr-1):
            r0  = rArr[ir]; r1 = rArr[ir+1]
            N   = np.where((R>=r0)*(R<r1))[0].size
            self.rbins[ir]  = N#/np.pi/(r1**2-r0**2)
            self.area[ir]   = np.pi*(r1**2-r0**2)
        self.rArr = rArr[:-1]
        if plotfig:
            plt.plot(self.rArr, self.rbins, 'o', ms=3)
            plt.show()
        self.r2     = (rArr[1:])**2-(rArr[:-1])**2
        self.rbins_norm =     self.rbins / self.r2
    
    def count_r_bins_cauchy_2d(self, Nt, gamma, rmax, Nr, rmin=0., plotfig=True):
        """
        Count data points for radius bins from 2D Cauchy distribution
        """
        rArr        = np.mgrid[rmin:rmax:Nr*1j]
        self.rArr   = rArr[:-1]
        pdf         = 1./np.pi/2.*(gamma/(((self.rArr)**2+gamma**2)**1.5) )
        self.rbins_cauchy = Nt * pdf * np.pi*((rArr[1:])**2-(rArr[:-1])**2)
        if plotfig:
            plt.plot(self.rArr, self.rbins_cauchy, 'o', ms=3)
            plt.show()
            
    def count_r_bins_gauss_2d(self, Nt, sigma, rmax, Nr, rmin=0., plotfig=True):
        """
        Count data points for radius bins from 2D Cauchy distribution
        """
        rArr        = np.mgrid[rmin:rmax:Nr*1j]
        self.rArr   = rArr[:-1]
        pdf         = np.exp( -0.5*((self.rArr/sigma)**2 )) / 2./np.pi/(sigma**2)
        self.rbins_gauss = Nt * pdf * np.pi*((rArr[1:])**2-(rArr[:-1])**2)
        if plotfig:
            plt.plot(self.rArr, self.rbins_cauchy, 'o', ms=3)
            plt.show()
    
    def count_r_bins_exprd_2d(self, Nt, C, rmax, Nr, rmin=0., plotfig=True):
        """
        Count data points for radius bins from 2D Cauchy distribution
        """
        rArr        = np.mgrid[rmin:rmax:Nr*1j]
        self.rArr   = rArr[:-1]
        pdf         = 1./2./np.pi/C*np.exp( -self.rArr/C)
        self.rbins_exprd = Nt * pdf * np.pi*((rArr[1:])**2-(rArr[:-1])**2)
        if plotfig:
            plt.plot(self.rArr, self.rbins_cauchy, 'o', ms=3)
            plt.show()
    
    def count_r_bins_mix_2d(self, Nt, p, sigma, C, rmax, Nr, rmin=0., plotfig=True):
        """
        Count data points for radius bins from 2D Cauchy distribution
        """
        rArr        = np.mgrid[rmin:rmax:Nr*1j]
        self.rArr   = rArr[:-1]
        # pdf1        = np.exp( -0.5*((self.rArr/sigma)**2 )) / 2./np.pi/(sigma**2)
        pdf1        = 1./np.pi/2.*(sigma/(((self.rArr)**2+sigma**2)**1.5) )
        pdf2        = 1./2./np.pi/C*np.exp( -self.rArr/C)
        self.rbins_mix = Nt*(p*pdf1+(1-p)*pdf2) * np.pi*((rArr[1:])**2-(rArr[:-1])**2)
        if plotfig:
            plt.plot(self.rArr, self.rbins_cauchy, 'o', ms=3)
            plt.show()
            
            
    def gauss_rbins_fit(self, plotfig=False, plotrmax=50, normtype=1):
        """
        Fit the 2D cross section of radius-binned data points with 1D Cauchy distribution
        """
        A0                      = self.rbins.max()
        sigmaArr                = np.arange(200)*.2 + .2 # double check
        # sigmaArr=np.array([30.])
        sigma_min, rms, Amin    = _gauss_rbins_fit(self.rArr, self.rbins, sigmaArr=sigmaArr, normtype=normtype)
        self.rbins_pre          = Amin/np.sqrt(2*np.pi)/sigma_min*np.exp(-(self.rArr)**2/2./(sigma_min**2))
        self.Amin               = Amin
        self.sigma_rbins        = sigma_min
        self.rms                = rms
        if plotfig:
            ax=plt.subplot()
            plt.plot(self.rArr, self.rbins, 'o', ms=10, label='observed')
            # plt.plot(self.rArr, self.rbins_pre_gauss, 'k--', lw=3, label='Best fit Gaussian distribution')
            plt.plot(self.rArr, self.rbins_pre, 'k--', lw=3, label='Best fit Gaussian distribution')
            plt.ylabel('PDF ', fontsize=30)
            plt.xlabel('Radius (nm)', fontsize=30)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            plt.legend(loc=0, fontsize=20, numpoints=1)
            plt.yscale('log', nonposy='clip')
            plt.xlim(0, plotrmax)
            plt.ylim(1e-8, 0.1)
            plt.show()
    
    def cauchy_rbins_fit(self, plotfig=False, plotrmax=50, normtype=1):
        """
        Fit the 2D cross section of radius-binned data points with 1D Cauchy distribution
        """
        A0                      = self.rbins.max()
        gammaArr                = np.arange(200)*.2 + .2 # double check
        Amin, gamma_min, rms, rmsArr = _cauchy_rbins_fit(self.rArr, self.rbins, A0=A0, gammaArr=gammaArr, normtype=normtype)
        self.rbins_pre          = Amin/np.pi*(gamma_min/((self.rArr)**2+gamma_min**2) )
        self.Amin               = Amin
        self.gamma_cauchy_rbins = gamma_min
        self.rms                = rms
        temp                    = gammaArr[rmsArr<rms*1.1]
        self.gmax    = temp.max(); self.gmin = temp.min()
        # print Amin, gamma_min, rms
        if plotfig:
            ax=plt.subplot()
            plt.plot(self.rArr, self.rbins, 'o', ms=10, label='observed')
            plt.plot(self.rArr, self.rbins_pre, 'k--', lw=3, label='Best fit Cauchy distribution')
            # plt.plot(self.rArr, self.rbins_pre_gauss, 'r--', lw=3, label='Best fit Gaussian distribution')
            plt.ylabel('PDF ', fontsize=30)
            plt.xlabel('Radius (nm)', fontsize=30)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            plt.legend(loc=0, fontsize=20, numpoints=1)
            plt.yscale('log', nonposy='clip')
            plt.xlim(0, plotrmax)
            plt.ylim(1e-8, 0.1)
            plt.show()
            
    def pdf_radius(self, ratio=0.5):
        NArr = self.rbins_pre*self.area
        A0 = NArr.sum()
        for i in xrange(NArr.size):
            temp = (NArr[:(i+1)]).sum()
            if temp/A0 > ratio:
                self.r_pdf = self.rArr[i]
                break
        return
            
            
    def get_char_radius(self, zmin=None, zmax=None, ratio=0.5):
        if zmin != None:
            if zmax == None: zmax = zmin + 10.
            ind     = (self.z >= zmin)*(self.z <= zmax)
            xin     = self.x[ind];  yin     = self.y[ind]; zin      = self.z[ind]
            self.z0 = (zmin+zmax)/2
        else:
            xin     = self.x.copy();  yin     = self.y.copy(); zin      = self.z.copy()
        rArr    = np.arange(400.)*2.+2.
        Nt      = xin.size
        for r in rArr:
            if (xin[(xin**2+yin**2)<=r**2]).size > Nt*ratio:
                try:
                    print 'z0 =',self.z0,' nm 1 sigma radius = ',r,' nm', ' Nin = ', (xin[(xin**2+yin**2)<=r**2]).size, 'Ntotal =', Nt
                except:
                    print '1 sigma radius = ',r,' nm', ' Nin = ', (xin[(xin**2+yin**2)<=r**2]).size, 'Ntotal =', Nt
                break
        self.cr = r
        return
    
    def select_char(self, energy, dE=1.):
        ind = (self.energy > energy-dE)*(self.energy < energy+dE)
        self.x = self.x[ind]; self.y = self.y[ind]; self.z = self.z[ind]
        self.u = self.u[ind]; self.v = self.v[ind]; self.w = self.w[ind]
        self.energy = self.energy[ind]
        return
        
    
    def cauchy_2d_fit(self, dN=200, dgamma=0.5, normtype=2):
        """
        Fit the 2D cross section of data points with Cauchy distribution
        """
        xgg, ygg= np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        print 'Finding minimum N and gamma'
        # Coarsest grid
        N0Arr   = np.arange(20)*5000. + 5000.
        gammaArr= np.arange(50)*2. + 2.
        Nmin, gamma_min, rms= _cauchy_2d_fit(xgg, ygg, self.nArr, N0Arr, gammaArr, normtype=normtype)
        # Coarsest grid
        N0Arr   = np.arange(10)*1000. + Nmin - 2500.
        gammaArr= np.arange(50)*1. + gamma_min - 1.
        Nmin, gamma_min, rms= _cauchy_2d_fit(xgg, ygg, self.nArr, N0Arr, gammaArr, normtype=normtype)
        # finest grid
        N0Arr   = np.arange(10)*dN + Nmin - 500.
        gammaArr= np.arange(50)*dgamma + gamma_min - 0.5
        Nmin, gamma_min, rms= _cauchy_2d_fit(xgg, ygg, self.nArr, N0Arr, gammaArr, normtype=normtype)
        self.Ncauchy        = Nmin
        self.gamma_cauchy   = gamma_min
        self.rms2d          = rms
        print 'End finding minimum N and gamma'
        print 'N =', Nmin,' gamma =', gamma_min 
        return
    
    def gauss_2d_fit(self, dN=200, dsigma=0.5, normtype=2):
        """
        Fit the 2D cross section of data points with Gauss distribution
        """
        xgg, ygg= np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        print 'Finding minimum N and gamma'
        # Coarsest grid
        N0Arr   = np.arange(20)*5000. + 5000.
        sigmaArr= np.arange(50)*2. + 2.
        Nmin, sigma_min, rms= _gauss_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, normtype=normtype)
        # Coarsest grid
        N0Arr   = np.arange(10)*1000. + Nmin - 2500.
        sigmaArr= np.arange(50)*1. + sigma_min - 1.
        Nmin, sigma_min, rms= _gauss_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, normtype=normtype)
        # finest grid
        N0Arr   = np.arange(10)*dN + Nmin - 500.
        sigmaArr= np.arange(50)*dsigma + sigma_min - 0.5
        Nmin, sigma_min, rms= _gauss_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, normtype=normtype)
        self.Ngauss         = Nmin
        self.sigma          = sigma_min
        self.rms2d          = rms
        print 'End finding minimum N and sigma'
        print 'N =', Nmin,' sigma =', sigma_min 
        return
    
    def cauchy_2d_fit_fix_N0(self, dN=200, dgamma=0.5, normtype=2):
        """
        Fit the 2D cross section of data points with Cauchy distribution
        """
        xgg, ygg= np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        N0 = self.nArr[(self.Nx-1)/2, (self.Ny-1)/2]
        gammaArr = np.arange(100)*dgamma + 0.5
        gamma_min, rms = _cauchy_2d_fit_fix_N0(xgg, ygg, self.nArr, N0=N0, gammaArr=gammaArr, normtype=normtype)
        self.gamma_cauchy   = gamma_min
        self.rms2d          = rms
        self.Ncauchy        = N0*2.*np.pi*(gamma_min**2)
        return
    
    def mix_2d_fit(self):
        """
        Fit the 2D cross section of data points with Gauss distribution
        """
        xgg, ygg= np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        print 'Finding minimum N and sigma'
        # Coarsest grid
        N0Arr   = np.arange(10)*5000. + 5000.
        sigmaArr= np.arange(5)*.1 + 3.8
        pArr    = np.arange(10)*.1 + .1
        # CArr    = 
        pmin, Cmin, Nmin, sigma_min, rms = _mix_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, pArr, CArr, normtype=normtype)
        # # Coarsest grid
        # N0Arr   = np.arange(10)*1000. + Nmin - 2500.
        # sigmaArr= np.arange(50)*1. + sigma_min - 1.
        # Nmin, sigma_min, rms= _gauss_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, normtype=normtype)
        # # finest grid
        # N0Arr   = np.arange(10)*dN + Nmin - 500.
        # sigmaArr= np.arange(50)*dsigma + sigma_min - 0.5
        # Nmin, sigma_min, rms= _gauss_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, normtype=normtype)
        # self.Ngauss         = Nmin
        # self.sigma          = sigma_min
        # self.rms2d          = rms
        print 'End finding minimum N and sigma'
        print 'N =', Nmin,' sigma =', sigma_min 
        return
    
    def exprd_2d_fit(self, normtype=2):
        """
        Fit the 2D cross section of data points with Gauss distribution
        """
        xgg, ygg= np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        print 'Finding minimum N and C'
        # Coarsest grid
        N0Arr   = np.arange(10)*5000. + 5000.
        CArr    = np.arange(50)*1. +1.
        Nmin, Cmin, rms = _exprd_2d_fit(xgg, ygg, self.nArr, N0Arr, CArr, normtype=normtype)
        # CArr    = 
        # pmin, Cmin, Nmin, sigma_min, rms = _mix_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, pArr, CArr, normtype=normtype)
        # # Coarsest grid
        # N0Arr   = np.arange(10)*1000. + Nmin - 2500.
        # sigmaArr= np.arange(50)*1. + sigma_min - 1.
        # Nmin, sigma_min, rms= _gauss_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, normtype=normtype)
        # # finest grid
        # N0Arr   = np.arange(10)*dN + Nmin - 500.
        # sigmaArr= np.arange(50)*dsigma + sigma_min - 0.5
        # Nmin, sigma_min, rms= _gauss_2d_fit(xgg, ygg, self.nArr, N0Arr, sigmaArr, normtype=normtype)
        self.Nexp           = Nmin
        self.C              = Cmin
        self.rms2d          = rms
        print 'End finding minimum N and C'
        print 'N =', Nmin,' sigma =', Cmin 
        return
    
    
    def plot_cauchy_2d_fit(self, showfig=True, outfname=None):
        plotx, ploty = np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        # 
        # # # #
        pdf     = _cauchy_2d(self.xgrid, self.ygrid, gamma=self.gamma_cauchy, mux=0., muy=0.)
        fig     = plt.figure(figsize=(12,8))
        
        ax      = plt.subplot(221)
        nArr_pre= pdf*self.Ncauchy
        Nmax    = self.nArr.max()
        plt.pcolormesh(plotx, ploty, nArr_pre, shading='gouraud', vmax=Nmax/2, vmin=0., cmap='hot_r')
        plt.xlabel('X (nm)', fontsize=10)
        plt.ylabel('Y (nm)', fontsize=10)
        plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        cb=plt.colorbar()
        cb.set_label('Number of photons', fontsize=10)

        ax = plt.subplot(222)
        plt.pcolormesh(plotx, ploty, self.nArr, shading='gouraud', vmax=Nmax/2, vmin=0., cmap='hot_r')
        plt.xlabel('X (nm)', fontsize=10)
        plt.ylabel('Y (nm)', fontsize=10)
        plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        cb=plt.colorbar()
        cb.set_label('Number of photons', fontsize=10)
        
        ax = plt.subplot(223)
        nyArr   = self.nArr[plotx==0.]
        nyArr2  = nArr_pre[plotx==0.]
        plt.plot(self.ygrid, nyArr, 'b-', lw=1, label='scatter, x = 0 nm')
        plt.plot(self.ygrid, nyArr2, 'k--', lw=2, label='best, x = 0 nm')
        # plt.xlim(-50, 50)
        plt.xlim(-100, 100)
        plt.legend(loc=0, fontsize=10)
        plt.title('X = 0 nm', fontsize=30)
        
        ax = plt.subplot(224)
        nxArr   = self.nArr[ploty==0.]
        nxArr2  = nArr_pre[ploty==0.]
        plt.plot(self.xgrid, nxArr, 'b-', lw=1, label='scatter, y = 0 nm')
        plt.plot(self.xgrid, nxArr2, 'k--', lw=2, label='best, y = 0 nm')

        plt.xlim(-100, 100)
        # plt.xlim(-50, 50)
        plt.title('Y = 0 nm', fontsize=30)
        plt.legend(loc=0, fontsize=10)
        if outfname !=None:
            plt.savefig(outfname, format='png')
        if showfig: plt.show()
        
        
    def plot_gauss_2d_fit(self, showfig=True, outfname=None):
        plotx, ploty = np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        # 
        # # # #
        pdf     = np.exp( -0.5*((plotx/self.sigma)**2) - 0.5*((ploty/self.sigma)**2) ) / 2./np.pi/(self.sigma**2)
        fig     = plt.figure(figsize=(12,8))
        
        ax      = plt.subplot(221)
        nArr_pre= pdf*self.Ngauss
        Nmax    = self.nArr.max()
        plt.pcolormesh(plotx, ploty, nArr_pre, shading='gouraud', vmax=Nmax/2, vmin=0., cmap='hot_r')
        plt.xlabel('X (nm)', fontsize=10)
        plt.ylabel('Y (nm)', fontsize=10)
        plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        cb=plt.colorbar()
        cb.set_label('Number of photons', fontsize=10)

        ax = plt.subplot(222)
        plt.pcolormesh(plotx, ploty, self.nArr, shading='gouraud', vmax=Nmax/2, vmin=0., cmap='hot_r')
        plt.xlabel('X (nm)', fontsize=10)
        plt.ylabel('Y (nm)', fontsize=10)
        plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        cb=plt.colorbar()
        cb.set_label('Number of photons', fontsize=10)
        
        ax = plt.subplot(223)
        nyArr   = self.nArr[plotx==0.]
        nyArr2  = nArr_pre[plotx==0.]
        plt.plot(self.ygrid, nyArr, 'b-', lw=1, label='scatter, x = 0 nm')
        plt.plot(self.ygrid, nyArr2, 'k--', lw=2, label='best, x = 0 nm')
        # plt.xlim(-50, 50)
        plt.xlim(-100, 100)
        plt.legend(loc=0, fontsize=10)
        plt.title('X = 0 nm', fontsize=30)
        
        ax = plt.subplot(224)
        nxArr   = self.nArr[ploty==0.]
        nxArr2  = nArr_pre[ploty==0.]
        plt.plot(self.xgrid, nxArr, 'b-', lw=1, label='scatter, y = 0 nm')
        plt.plot(self.xgrid, nxArr2, 'k--', lw=2, label='best, y = 0 nm')

        plt.xlim(-100, 100)
        # plt.xlim(-50, 50)
        plt.title('Y = 0 nm', fontsize=30)
        plt.legend(loc=0, fontsize=10)
        if outfname !=None:
            plt.savefig(outfname, format='png')
        if showfig: plt.show()
    
    def get_cauchy_e_width(self):
        pdf     = _cauchy_2d(self.xgrid, self.ygrid, gamma=self.gamma_cauchy, mux=0., muy=0.)
        
    def hist(self, repeat=6250, bins=7500, plotfig=False):
        ax=plt.subplot(211)
        n, bins, patches = plt.hist(np.repeat(self.energy, repeat), bins=bins, normed=False, facecolor='blue', edgecolor='blue')
        if plotfig:
            plt.ylabel('Photons/sec', fontsize=30)
            # plt.xlabel('Energy (eV)', fontsize=30)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            plt.title('Input Spectrum', fontsize=40)
            plt.yscale('log', nonposy='clip')
            plt.ylim(1e4, 1e8)
            dbins=bins[0] - bins[1]
            plt.grid(True)
        else: plt.close()
        self.ebins  = (bins[1:] + bins[:-1])/2 # convert to keV
        self.Np     = n 
        return
    
    def decay_spec(self, elesym, t, plotfig=True, density=None):
        t   = t/1e7
        if density == None:
            mu  = (xraylib_func.get_mu_np(energy=self.ebins/1000., elesym=elesym))[0,:]
        else:
            mu  = xraylib_func.get_mu_np_CP(energy=self.ebins/1000., chemicalform=elesym, density=density)
        r   = np.exp(-mu*t)
        self.Np_out= r*self.Np
        if plotfig:
            ax=plt.subplot()
            plt.plot(self.ebins, self.Np_out)
            plt.ylabel('Photons/sec', fontsize=30)
            plt.xlabel('Energy (eV)', fontsize=30)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            plt.yscale('log', nonposy='clip')
            plt.ylim(1e4, 1e8)
            plt.grid(True)
            plt.show()
        
        
    # 
    # def load(self, infname):
    #     """
    #     Load ASDF file
    #     """
    #     self.tree.update((asdf.AsdfFile.open(infname)).tree)
    
    
# def compute_gamma(elementnumber, energy=10):
    
    
    