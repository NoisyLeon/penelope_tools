import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
import numba


def get_polar_angles(u, v, w):
    """
    Define a wave propagation vector in a Cartesian coordinate.
    It is always explicitly normalized to lie on the unit sphere.
    ============================================================================
    Input Parameters:
    pv              - wave propagation direction vector
                        list of 3 numbers or numpy array (e.g. [1, 0, 0])
    is_normalized   - whether the pv is unit vector or not
    ============================================================================
    """
    theta = np.zeros(u.size); phi = np.zeros(u.size)
    ind_neg = w == -1.0; ind_pos = w == 1.0
    ind_else = np.logical_not(ind_neg*ind_pos)
    theta[ind_neg] = np.pi; theta[ind_pos] = 0.
    phi[ind_neg*ind_pos] = 0.
    theta[ind_else] = np.arccos(w[ind_else])
    sin_theta       = np.sqrt(1 - w[ind_else]**2)
    cos_phi         = u[ind_else]/sin_theta
    phi[ind_else]   = np.arccos(cos_phi)
    phi[v<0.]       = 2.0*np.pi - phi[v<0.]
    theta           = theta*180./np.pi
    phi             = phi*180./np.pi
    return theta, phi
    # 
    # self.clear_direction()
    # if not isinstance(pv, np.ndarray): pv = np.asarray(pv)
    # x, y, z = pv
    # if not is_normalized:
    #     n = np.sqrt(x*x + y*y + z*z)
    #     x = x/n
    #     y = y/n
    #     z = z/n
    # if z == 1.0 or z == -1.0:
    #     if z > 0.0: self.theta = 0.0
    #     else: self.theta = np.pi
    #     self.phi = 0.0
    # else:
    #     self.theta  = np.arccos(z)
    #     sin_theta   = np.sqrt(1 - z**2)
    #     cos_phi     = x/sin_theta
    #     self.phi    = np.arccos(cos_phi)
    #     if y < 0.0: self.phi = 2.0*np.pi - self.phi
    # self.theta      = self.theta*180./np.pi
    # self.phi        = self.phi*180./np.pi
    # self.pv         = pv/n
    # self.get_kc_mat()

def count_points(nArr, thetag, phig, thetain, phiin, dx):
    for itheta in xrange(thetag.size):
        for iphi in xrange(phig.size):
            theta = thetag[itheta]; phi = phig[iphi]
            tmin = theta - dx; tmax = theta + dx
            pmin = phi - dx; pmax = phi + dx
            N=np.where((thetain>=tmin)*(thetain<tmax)*(phiin>=pmin)*(phiin<pmax))[0].size
            nArr[itheta, iphi] = N
    return nArr

def count_points_2(nArr, zg, phig, zin, phiin, dphi, dz):
    for iz in xrange(zg.size):
        for iphi in xrange(phig.size):
            z = zg[iz]; phi = phig[iphi]
            zmin = z - dz; zmax = z + dz
            pmin = phi - dphi; pmax = phi + dphi
            N=np.where((zin>=zmin)*(zin<zmax)*(phiin>=pmin)*(phiin<pmax))[0].size
            nArr[iz, iphi] = N
    return nArr


pfx = 'I_1nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
u = inArr[:, 5]; v = inArr[:, 6]; w = inArr[:, 7]

theta, phi = get_polar_angles(u, v, w)




# ax = plt.subplot(projection='3d')
# ax.plot(u, v, w, 'o', ms=0.1)
# 
# 
# ax.set_xlim((-1, 1))
# ax.set_ylim((-1, 1))
# ax.set_zlim((-1, 1))
# 
# 

#########################
 
thetag = np.arange(181); phig = np.arange(360)
thetap, phip = np.meshgrid(thetag, phig, indexing='ij')
nArr  = np.zeros(thetap.shape)

nArr    = count_points(nArr, thetag, phig, theta, phi, dx=.5)

fig     = plt.figure(figsize=(12,8))
# ax      = plt.subplot(221)
plt.pcolormesh(thetap, phip, nArr, shading='gouraud', vmax=nArr.max(), vmin=0., cmap='hot_r')
plt.xlabel('theta (deg)', fontsize=30)
plt.ylabel('phi (deg)', fontsize=30)
# plt.axis([-10000, 10000, -10000, 10000], 'scaled')
cb=plt.colorbar()
# cb.set_label('%', fontsize=10)

# ##############
# zg = np.arange(201)*0.01 - 1.; phig = np.arange(360)
# zp, phip = np.meshgrid(zg, phig, indexing='ij')
# nArr  = np.zeros(zp.shape)
# nArr    = count_points_2(nArr, zg, phig, w, phi, dphi=.5, dz=0.005)
# 
# fig     = plt.figure(figsize=(12,8))
# # ax      = plt.subplot(221)
# plt.pcolormesh(zp, phip, nArr, shading='gouraud', vmax=nArr.max(), vmin=0., cmap='hot_r')
# plt.xlabel('z ', fontsize=30)
# plt.ylabel('phi (deg)', fontsize=30)
# # plt.axis([-10000, 10000, -10000, 10000], 'scaled')
# cb=plt.colorbar()
# #############

plt.show()


