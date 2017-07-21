import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal
import numba


def set_direction_cartesian(u, v, w):
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
    return


pfx = 'I_1nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
inArr = np.loadtxt(infname)
x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
u = inArr[:, 5]; v = inArr[:, 6]; w = inArr[:, 7]

theta, phi = set_direction_cartesian(u, v, w)


# 
# plst = ['I_1nA_E_30keV','I_5nA_E_30keV',  'I_10nA_E_30keV', 'I_100nA_E_30keV']
# 
# # plst = ['I_1nA_E_30keV']
# ax=plt.subplot()
# clst=['k', 'b', 'r', 'g']
# i=0
# for pfx in plst:
#     # pfx = 'I_1nA_E_30keV'
#     dArr    = np.array([])
#     zArr    = np.arange(19)*5.
#     for zmin in zArr:
#         infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
#         inArr = np.loadtxt(infname)
#         x = inArr[:, 2]*1e7; y = inArr[:, 3]*1e7; z = inArr[:, 4]*1e7
#         
#         zmax = zmin + 10
#         ind_valid = (z >= zmin)*(z <= zmax)
#         xin = x[ind_valid]; yin = y[ind_valid]; zin = z[ind_valid]
#         rArr    = np.arange(2000.)*0.05+0.05
#         z0      = (zmin+zmax)/2
#         Nt      = xin.size
#         for r in rArr:
#             if (xin[(xin**2+yin**2)<=r**2]).size > Nt*0.5:
#                 print 'z0 =',z0,' nm 1 sigma radius = ',r,' nm', ' Nin = ', (xin[(xin**2+yin**2)<=r**2]).size, 'Ntotal =', Nt
#                 break
#         
# #     
#         dArr    = np.append(dArr, 2.*r)
#     zArr    = zArr +5.
#     plt.plot(zArr, dArr, clst[i]+'o--', lw=3, ms=8, label=pfx)
#     i+=1
# plt.legend(loc=0, fontsize=20)
# plt.ylabel('Characteristic Diameter', fontsize=30)
# plt.xlabel('Z (nm)', fontsize=30)
# # plt.title('1 sigma (68 %)', fontsize=40)
# plt.title('half (50 %)', fontsize=40)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# plt.xlim(0., 100.)
# plt.show()


