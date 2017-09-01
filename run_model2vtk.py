import chipmodel

m=chipmodel.chipmodel(Nx=333, Ny=333, Nz=10, dx=30, dy=30, dz=1000, xmin=-332/2*30, ymin=-332/2*30)
m.load_mat_data(infname='ex1.mat', label='Y')
# m.write_structured_vtk(outfname='ex2.vtk')
m.write_structured_vtk_layerZ(outdir='ex1_model')