import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal

plst = ['I_1nA_E_30keV', 'I_10nA_E_30keV', 'I_100nA_E_30keV']
for pfx in plst:
    # pfx = 'I_1nA_E_30keV'
    for zmin in [0., 4e-6, 9e-6]:
        infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
        inArr = np.loadtxt(infname)
        x = inArr[:, 2]; y = inArr[:, 3]; z = inArr[:, 4]
        
        # ind_valid = (x >= -1e-5)*(x <= 1e-5) * (y >= -1e-5)*(y <= 1e-5) * (z >= 0.)*(z <= 1e-5)
        
        zmax = zmin + 1e-6
        ind_valid = (z >= zmin)*(z <= zmax)
        xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]
        # 
        xy      = np.vstack([xin,yin])
        
        # do kernel density estimation to get smooth estimate of distribution
        # make grid of points
        # 
        N=201j
        print 'computing PDF'
        kernel      = scipy.stats.gaussian_kde(xy, bw_method=None)
        x, y        = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
        positions   = np.vstack((x.ravel(), y.ravel()))
        density     = np.reshape(kernel.pdf(positions).T, x.shape)
        print 'End computing PDF'
        
        
        
        pos     = np.dstack((x, y))
        COV     = np.diag(kernel.covariance.diagonal())
        rv      = multivariate_normal(mean=[0, 0], cov=COV)
        zpdf2   = rv.pdf(pos)
        zpdf2   = zpdf2 / zpdf2.sum()*100.
        vmax=np.abs(zpdf2).max()
        # 
        # plot points
        # plot projection of density onto z-axis
        zpdf = density / density.sum()*100.
        plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
        #
        fig     = plt.figure(figsize=(12,8))
        ax      = plt.subplot(221)
        plt.pcolormesh(plotx, ploty, zpdf, shading='gouraud', vmax=vmax, vmin=0.)
        plt.xlabel('X (cm)', fontsize=10)
        plt.ylabel('Y (cm)', fontsize=10)
        plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'scaled')
        cb=plt.colorbar()
        cb.set_label('%', fontsize=10)
        plt.title('CDF from KDE')
        
        # # 
        # 
        # pos     = np.dstack((x, y))
        # COV     = np.diag(kernel.covariance.diagonal())
        # rv      = multivariate_normal(mean=[0, 0], cov=COV)
        ax      = plt.subplot(222)
        # zpdf2   = rv.pdf(pos)
        # zpdf2   = zpdf2 / zpdf2.sum()*100.
        plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
        plt.pcolormesh(plotx, ploty, zpdf2, shading='gouraud', vmax=vmax, vmin=0.)
        # 
        plt.xlabel('X (cm)', fontsize=10)
        plt.ylabel('Y (cm)', fontsize=10)
        plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'scaled')
        cb=plt.colorbar()
        cb.set_label('%', fontsize=10)
        plt.title('CDF from best fitting Gaussian distribution')
        
        diff_zpdf = (zpdf - zpdf2)/zpdf.max()
        # 
        ax = plt.subplot(223)
        plotx, ploty = np.mgrid[-1e-5:1e-5:N, -1e-5:1e-5:N]
        vmax=np.abs(diff_zpdf).max()
        plt.pcolormesh(plotx, ploty, diff_zpdf, shading='gouraud', cmap='seismic', vmax=vmax, vmin=-vmax)
        # 
        plt.title('')
        plt.xlabel('X (cm)', fontsize=10)
        plt.ylabel('Y (cm)', fontsize=10)
        plt.axis([-1e-5, 1e-5, -1e-5, 1e-5], 'equal')
        cb=plt.colorbar()
        # cb.set_label('%', fontsize=25)
        
        z0 = (zmin+zmax)/2
        plt.suptitle(pfx + ' z = ' +str(z0)+' cm', fontsize=20)
        print kernel.covariance
        
        outdir = '/home/leon/pdf_2D'
        outfname = outdir+'/'+pfx+'_'+str(z0)+'.pdf'
    
        # ax = plt.subplot(224)
        # plt.text(0, 0.1, np.sqrt(kernel.covariance.diagonal()), fontsize=15)
        # plt.text(0, 0.3, kernel.covariance, fontsize=15)
        ax = plt.subplot(224)
        xpdf = zpdf[ploty==0.]
        xpdf2= zpdf2[ploty==0.]
        
        ypdf = zpdf[plotx==0.]
        ypdf2= zpdf2[plotx==0.]
        # 
        plt.plot(np.mgrid[-1e-5:1e-5:N],xpdf, 'r-', lw=3, label='KDE, y = 0 cm')
        plt.plot(np.mgrid[-1e-5:1e-5:N], xpdf2, 'b-', lw=3, label='best, y = 0 cm')
        
        plt.plot(np.mgrid[-1e-5:1e-5:N],ypdf, 'k--', lw=3, label='KDE, x = 0 cm')
        plt.plot(np.mgrid[-1e-5:1e-5:N], ypdf2, 'g--', lw=3, label='best, x = 0 cm')
        plt.xlim(-1e-5, 1e-5)
        plt.legend(loc=0, fontsize=10)
        plt.savefig(outfname, format='pdf')
    # plt.show()



