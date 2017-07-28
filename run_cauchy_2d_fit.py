import database

pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
dbase.count_2d_points(zmin=40)
# dbase.cauchy_2d_fit(zmin=90.)
dbase.Ncauchy = 4400.
dbase.gamma_cauchy = 9.5
dbase.plot_cauchy_2d_fit()