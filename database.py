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


def _cauchy_2d_fit(xgg, ygg, nArr, N0Arr, gammaArr):
    rms = 999
    gamma_min = 0.
    N_min   = 0.
    for N in N0Arr:
        # N   = iN*100 + 26000
        for gamma in gammaArr:
            # gamma = ig*0.5 + 45.
            # print N, gamma
            pdf         = 1./np.pi/2.*(gamma/(((xgg)**2+(ygg)**2+gamma**2)**1.5) )
            nArr_pre    = pdf*N
            temp        = (nArr_pre - nArr)**2
            temp        = temp[nArr>0]
            # print temp.max()
            rms_temp    = np.sqrt(np.mean(temp))
            if rms_temp < rms:
                rms = rms_temp; gamma_min = gamma; Nmin = N
        # print N, gamma_min, rms
    return Nmin, gamma_min, rms

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
    
    
    
    # for sigma in sigmaArr:
    #     Aarr = A0*np.pi*gamma/2. + np.arange(100.)*A0*np.pi*gamma/100.
    #     rms_g= 1e9
    #     for A in Aarr:
    #         pdf     = A/np.pi*(gamma/((rg)**2+gamma**2) )
    #         if normtype == 2:
    #             temp    = (pdf - rbins)**2
    #             temp    = temp[rbins>0]
    #             rms_temp= np.sqrt(np.mean(temp))
    #         elif normtype == 1:
    #             temp    = np.abs(pdf - rbins)
    #             temp    = temp[rbins>0]
    #             rms_temp= np.mean(temp)
    #         else:
    #             raise ValueError('Not supported norm type: '+str(normtype))
    #         if rms_temp < rms:
    #             rms = rms_temp; gamma_min = gamma; Amin = A
    #         if rms_temp < rms_g:
    #             rms_g = rms_temp
    #     rmsArr= np.append(rmsArr, rms_g)
    # return Amin, gamma_min, rms, rmsArr

def _cauchy_2d(xg, yg, gamma, mux, muy):
    xgg, ygg = np.meshgrid(xg, yg, indexing='ij')
    pdf = 1./np.pi/2.*(gamma/(((xgg-mux)**2+(ygg-muy)**2+gamma**2)**1.5) )
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
    
    def count_2d_points(self, zmin, zmax=None):
        """
        Count data points for predefined grid
        """
        if zmax == None: zmax = zmin + 10.
        dx          = (self.xgrid[1] - self.xgrid[0])/2.
        ind         = (self.z >= zmin)*(self.z <= zmax)
        xin = self.x[ind]; yin = self.y[ind]; zin = self.z[ind]
        nArr        = np.zeros((self.xgrid.size, self.ygrid.size), np.int32)
        self.z2d   = (zmin+zmax)/2.
        print 'Counting 2D points for z =', self.z2d,' nm'
        self.nArr   = _count_2D_points(nArr, self.xgrid, self.ygrid, xin, yin, dx)
        print 'End Counting 2D points.'
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
        self.rbins_no_norm=self.rbins.copy()
        # self.rbins= self.rbins/self.rbins.sum()
        if plotfig:
            plt.plot(self.rArr, self.rbins, 'o', ms=3)
            plt.show()
            
            
    def gauss_rbins_fit(self, plotfig=False, plotrmax=50, normtype=1):
        """
        Fit the 2D cross section of radius-binned data points with 1D Cauchy distribution
        """
        A0                      = self.rbins.max()
        sigmaArr                = np.arange(200)*.2 + .2 # double check
        # sigmaArr=np.array([30.])
        sigma_min, rms, Amin    = _gauss_rbins_fit(self.rArr, self.rbins, sigmaArr=sigmaArr, normtype=normtype)
        # # # sigma_min = 25.
        # self.rbins_pre_gauss    = Amin/np.sqrt(2*np.pi)/sigma_min*np.exp(-(self.rArr)**2/2./(sigma_min**2))
        self.rbins_pre          = Amin/np.sqrt(2*np.pi)/sigma_min*np.exp(-(self.rArr)**2/2./(sigma_min**2))
        self.Amin               = Amin
        self.sigma_rbins        = sigma_min
        self.rms                = rms
        
        # temp                    = gammaArr[rmsArr<rms*1.1]
        # self.gmax    = temp.max(); self.gmin = temp.min()
        # print Amin, gamma_min, rms
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
        
    
    def cauchy_2d_fit(self, zmin, zmax=None, dN=200, dgamma=0.5):
        """
        Fit the 2D cross section of data points with Cauchy distribution
        """
        if zmax == None: zmax = zmin + 10.
        xgg, ygg= np.meshgrid(self.xgrid, self.ygrid, indexing='ij')
        print 'Finding minimum N and gamma'
        # Coarsest grid
        N0Arr   = np.arange(10)*5000. + 5000.
        gammaArr= np.arange(50)*2. + 2.
        Nmin, gamma_min, rms= _cauchy_2d_fit(xgg, ygg, self.nArr, N0Arr, gammaArr)
        # Coarsest grid
        N0Arr   = np.arange(10)*1000. + Nmin - 2500.
        gammaArr= np.arange(50)*1. + gamma_min - 1.
        Nmin, gamma_min, rms= _cauchy_2d_fit(xgg, ygg, self.nArr, N0Arr, gammaArr)
        # finest grid
        N0Arr   = np.arange(10)*dN + Nmin - 500.
        gammaArr= np.arange(50)*dgamma + gamma_min - 0.5
        Nmin, gamma_min, rms= _cauchy_2d_fit(xgg, ygg, self.nArr, N0Arr, gammaArr)
        self.Ncauchy        = Nmin
        self.gamma_cauchy   = gamma_min
        self.rms2d          = rms
        print 'End finding minimum N and gamma'
        print 'N =', Nmin,' gamma =', gamma_min 
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
    
    
    