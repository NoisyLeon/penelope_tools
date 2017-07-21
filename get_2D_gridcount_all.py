import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
import numba

def count_points(nArr, xg, yg, xin, yin, dx):
    for ix in xrange(xg.size):
        for iy in xrange(yg.size):
            x   = xg[ix]; y = yg[iy]
            xmin = x - dx; xmax = x + dx
            ymin = y - dx; ymax = y + dx
            N=np.where((xin>=xmin)*(xin<xmax)*(yin>=ymin)*(yin<ymax))[0].size
            nArr[ix, iy] = N
    return nArr


plst = ['I_1nA_E_30keV','I_5nA_E_30keV',  'I_10nA_E_30keV', 'I_100nA_E_30keV']

# plst = ['I_1nA_E_30keV']
ax=plt.subplot()
clst=['k', 'b', 'r', 'g']
i=0
for pfx in plst:
    # pfx = 'I_1nA_E_30keV'
    dArr    = np.array([])
    zArr    = np.arange(19)*5.
    for zmin in zArr:
        infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
        inArr = np.loadtxt(infname)
        x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
        
        zmax = zmin + 10
        ind_valid = (z >= zmin)*(z <= zmax)
        xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]
        
        N       = 201j
        xg      = np.mgrid[-100:100:N]
        yg      = np.mgrid[-100:100:N]
        nArr    = np.zeros((xg.size, yg.size))
        
        nArr    = count_points(nArr, xg, yg, xin, yin, dx=(xg[1]-xg[0])/2.)
        Nt      = nArr.sum()
        rArr    = np.arange(1000.)*0.1+0.1
        plotx, ploty = np.mgrid[-100:100:N, -100:100:N]
        z0      = (zmin+zmax)/2
        for r in rArr:
            if (nArr[(plotx**2+ploty**2)<=r**2]).sum() > Nt*0.95:
                print 'z0 =',z0,' nm half radius = ',r,' nm', ' Nin = ', nArr[(plotx**2+ploty**2)<r**2].sum(), 'Ntotal =', Nt
                # print nArr[(plotx**2+ploty**2)<=(r-40.)**2].sum()
                break
        
        pdf     = nArr/Nt
        xpdf    = pdf[100, :]
        ypdf    = pdf[:, 100]
        plotx, ploty = np.mgrid[-100:100:N, -100:100:N]
        
        # 
        # # #
        # fig     = plt.figure(figsize=(12,8))
        # ax      = plt.subplot(221)
        # plt.pcolormesh(plotx, ploty, pdf, shading='gouraud', vmax=pdf.max(), vmin=0.)
        # plt.xlabel('X (nm)', fontsize=10)
        # plt.ylabel('Y (nm)', fontsize=10)
        # plt.axis([-100, 100, -100, 100], 'scaled')
        # cb=plt.colorbar()
        # cb.set_label('%', fontsize=10)
        # 
        # 
        # ax = plt.subplot(222)
        # plt.text(0.2, 0.5, 'half radius = '+str(r)+' nm', fontsize=15)
        # 
        # ax = plt.subplot(224)
        # # 
        # plt.plot(xg, xpdf, 'r-', lw=3, label='PDF, y = 0 nm')
        # plt.plot(yg, ypdf, 'k--', lw=3, label='PDF, x = 0 nm')
        # plt.xlim(-100, 100)
        # plt.legend(loc=0, fontsize=10)
        # z0 = (zmin+zmax)/2
        # plt.suptitle(pfx + ' z = ' +str(z0)+' nm', fontsize=20)
        # 
        # 
        # outdir = '/home/leon/pdf_2D'
        # z0=(zmin+zmax)/2.
        # outfname = outdir+'/'+pfx+'_'+str(z0)+'nm.pdf'
        # plt.savefig(outfname, format='pdf')
    
        dArr    = np.append(dArr, 2.*r)
    zArr    = zArr +5.
    plt.plot(zArr, dArr, clst[i]+'o--', lw=3, ms=8, label=pfx)
    i+=1
plt.legend(loc=0, fontsize=20)
plt.ylabel('Characteristic Diameter', fontsize=30)
plt.xlabel('Z (nm)', fontsize=30)
plt.title('2 sigma (95%)', fontsize=40)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.xlim(0., 100.)
plt.show()


