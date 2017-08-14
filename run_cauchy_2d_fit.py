import database

pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
# dbase.count_2d_points(zmin=None, outfname='all.npy')
dbase.load_2d_points(infname='all.npy')

# dbase.cauchy_2d_fit_fix_N0(normtype=.5)

dbase.exprd_2d_fit(normtype=.5)
# dbase.gauss_2d_fit(normtype=2.)
# dbase.Ncauchy = 4400.
# dbase.gamma_cauchy = 9.5
# dbase.plot_cauchy_2d_fit()
# dbase.plot_gauss_2d_fit()