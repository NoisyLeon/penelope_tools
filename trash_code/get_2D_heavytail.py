import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
import numba
from numba import float64, int32
import seaborn as sns
from math import *

def multivariate_t_distribution(x,mu,Sigma,df,d):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    '''
    Num = gamma(1. * (d+df)/2)
    Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(np.linalg.det(Sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), (x - mu)),1.* (d+df)/2))
    d = 1. * Num / Denom 
    return d

@numba.jit(int32[:,:](int32[:,:], float64[:], float64[:], float64[:], float64[:], float64) )
# @numba.jit()
def count_points(nArr, xg, yg, xin, yin, dx):
    for ix in xrange(xg.size):
        for iy in xrange(yg.size):
            x   = xg[ix]; y = yg[iy]
            xmin = x - dx; xmax = x + dx
            ymin = y - dx; ymax = y + dx
            N=np.where((xin>=xmin)*(xin<xmax)*(yin>=ymin)*(yin<ymax))[0].size
            nArr[ix, iy] = N
    return nArr

def cauchy_2D(xg, yg, gamma, mux, muy):
    xgg, ygg = np.meshgrid(xg, yg, indexing='ij')
    pdf = 1./np.pi/2.*(gamma/(((xgg-mux)**2+(ygg-muy)**2+gamma**2)**1.5) )
    return pdf
    

pfx = 'I_1nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7

zmin = 0; zmax = zmin + 10
ind_valid = (z >= zmin)*(z <= zmax)
xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]
# print xin.min(), xin.max(), yin.min(), yin.max()
N       = 1601j
xg      = np.mgrid[-800:800:N]
yg      = np.mgrid[-800:800:N]
nArr    = np.zeros((xg.size, yg.size), np.int32)

nArr    = count_points(nArr, xg, yg, xin, yin, dx=(xg[1]-xg[0])/2.)
Nt      = xin.size

##
# Cauchy distribution
##
# from scipy.stats import cauchy

plotx, ploty = np.meshgrid(xg, yg, indexing='ij')
Nt = nArr.sum()
Nmax=nArr.max()
rms = 999.
gamma_min = 0.
print Nt, Nmax, float(Nmax/Nt)
# for gamma in np.arange(30.)*0.1+.1:
for gamma in np.arange(100.)*0.5+.5:
    pdf = cauchy_2D(xg, yg, gamma=gamma, mux=0., muy=0.)
    nArr_pre = pdf*Nt
    rms_temp = np.sqrt(np.mean((nArr_pre - nArr)**2))
    # print rms_temp, gamma
    if rms_temp < rms:
        rms = rms_temp; gamma_min = gamma
    
print gamma_min, rms
##
# Student's t distribution
##
# from scipy.stats import t

# 
# pdf     = nArr/Nt
# xpdf    = pdf[100, :]
# ypdf    = pdf[:, 100]
plotx, ploty = np.meshgrid(xg, yg, indexing='ij')
# 
# # # #
pdf = cauchy_2D(xg, yg, gamma=gamma_min, mux=0., muy=0.)
fig     = plt.figure(figsize=(12,8))
ax      = plt.subplot(221)
nArr_pre = pdf*Nt
plt.pcolormesh(plotx, ploty, nArr_pre, shading='gouraud', vmax=Nmax/2, vmin=0., cmap='hot_r')
plt.xlabel('X (nm)', fontsize=10)
plt.ylabel('Y (nm)', fontsize=10)
plt.axis([-800, 800, -800, 800], 'scaled')
cb=plt.colorbar()
cb.set_label('Number of photons', fontsize=10)
# # 
# # 
ax = plt.subplot(222)
plt.pcolormesh(plotx, ploty, nArr, shading='gouraud', vmax=Nmax/2, vmin=0., cmap='hot_r')
plt.xlabel('X (nm)', fontsize=10)
plt.ylabel('Y (nm)', fontsize=10)
plt.axis([-800, 800, -800, 800], 'scaled')
cb=plt.colorbar()
cb.set_label('Number of photons', fontsize=10)

ax = plt.subplot(223)
# g = sns.jointplot(xin, yin, kind="scatter", size=7, space=0, xlim={-800, 800}, ylim={-800, 800})
# plt.pcolormesh(plotx, ploty, nArr, shading='gouraud', vmax=2, vmin=0., cmap='hot_r')
# plt.scatter(xin, yin, alpha=0.5)
# plt.xlabel('X (nm)', fontsize=10)
# plt.ylabel('Y (nm)', fontsize=10)
# plt.axis([-800, 800, -800, 800], 'scaled')
# cb=plt.colorbar()
# cb.set_label('Number of photons', fontsize=10)
nyArr   = nArr[plotx==0.]
nyArr2  = nArr_pre[plotx==0.]
plt.plot(np.mgrid[-800:800:N],nyArr, 'b-', lw=1, label='scatter, x = 0 nm')
plt.plot(np.mgrid[-800:800:N], nyArr2, 'k--', lw=2, label='best, x = 0 nm')
plt.xlim(-50, 50)
# plt.xlim(-100, 100)
plt.legend(loc=0, fontsize=10)
plt.title('X = 0 nm', fontsize=30)

ax = plt.subplot(224)
nxArr   = nArr[ploty==0.]
nxArr2  = nArr_pre[ploty==0.]
# 
plt.plot(np.mgrid[-800:800:N], nxArr, 'b-', lw=1, label='scatter, y = 0 nm')
plt.plot(np.mgrid[-800:800:N], nxArr2, 'k--', lw=2, label='best, y = 0 nm')
# 
# plt.plot(np.mgrid[-800:800:N],nyArr, 'k--', lw=1, label='scatter, x = 0 nm')
# plt.plot(np.mgrid[-800:800:N], nyArr2, 'g--', lw=1, label='best, x = 0 nm')
# plt.xlim(-100, 100)
plt.xlim(-50, 50)
plt.title('Y = 0 nm', fontsize=30)
plt.legend(loc=0, fontsize=10)
plt.show()



