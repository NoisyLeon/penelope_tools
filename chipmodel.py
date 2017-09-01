import numpy as np
import scipy.io
import os

class chipmodel(object):
    
    def __init__(self, Nx, dx, Ny, dy, Nz, dz, xmin=0., ymin=0., zmin=0.):
        self.Nx     = Nx
        self.Ny     = Ny
        self.Nz     = Nz
        self.xgrid  = np.arange(Nx)*dx+xmin
        self.ygrid  = np.arange(Ny)*dy+ymin
        self.zgrid  = np.arange(Nz)*dz+zmin
        return
    
    def load_mat_data(self, infname, label='Y'):
        indat       = scipy.io.loadmat(infname)
        self.data   = indat[label]
        return
    
    def write_structured_vtk(self, outfname, dataname='scalars'):
        """
        v.shape = nx, ny, nz
        """
        N = self.Nx*self.Ny*self.Nz # total number of grid
        with open(outfname, 'wb') as fid:
            # header information
            fid.write('# vtk DataFile Version 4.0\n')
            fid.write('vtk output\n')
            fid.write('ASCII\n')
            fid.write('DATASET STRUCTURED_GRID\n')
            fid.write('DIMENSIONS %d %d %d' %(self.Nx, self.Ny, self.Nz))
            fid.write('POINTS '+str(N)+' float\n')
            # grid coordinate
            for k in xrange(self.Nz):
                for j in xrange(self.Ny):
                    for i in xrange(self.Nx):
                        fid.write(str(self.xgrid[i])+' '+str(self.ygrid[j])+' '+str(self.zgrid[k])+'\n')
            #- write data
            fid.write('\n')
            fid.write('POINT_DATA '+str(N)+'\n')
            fid.write('SCALARS ' +dataname+' float\n')
            fid.write('LOOKUP_TABLE default\n')
            for k in xrange(self.Nz):
                for j in xrange(self.Ny):
                    for i in xrange(self.Nx):
                        fid.write(str(self.data[i,j,k])+'\n')
        return
    
    def write_structured_vtk_layerZ(self, outdir, dataname='scalars', zspacing=None):
        """
        v.shape = nx, ny, nz
        """
        if zspacing==None: zspacing=self.zgrid[1] - self.zgrid[0]
        if not os.path.isdir(outdir): os.makedirs(outdir)
        N = self.Nx*self.Ny # total number of grid
        for iz in xrange(self.Nz):
            outfname = outdir+'/'+str(iz)+'.vtk'
            with open(outfname, 'wb') as fid:
                # header information
                fid.write('# vtk DataFile Version 4.0\n')
                fid.write('vtk output\n')
                fid.write('ASCII\n')
                fid.write('DATASET STRUCTURED_GRID\n')
                fid.write('DIMENSIONS %d %d %d' %(self.Nx, self.Ny, 1))
                fid.write('POINTS '+str(N)+' float\n')
                # grid coordinate
                for k in xrange(1):
                    for j in xrange(self.Ny):
                        for i in xrange(self.Nx):
                            if iz !=0:
                                fid.write(str(self.xgrid[i])+' '+str(self.ygrid[j])+' '+str(self.zgrid[iz]+zspacing)+'\n')
                            else:
                                fid.write(str(self.xgrid[i])+' '+str(self.ygrid[j])+' '+str(self.zgrid[iz])+'\n')
                #- write data
                fid.write('\n')
                fid.write('POINT_DATA '+str(N)+'\n')
                fid.write('SCALARS ' +dataname+' float\n')
                fid.write('LOOKUP_TABLE default\n')
                for k in xrange(1):
                    for j in xrange(self.Ny):
                        for i in xrange(self.Nx):
                            fid.write(str(self.data[i,j,iz])+'\n')
        return
