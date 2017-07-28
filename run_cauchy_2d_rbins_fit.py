import database

pfx     = 'I_100nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)
dbase.count_r_bins(zmin=90., rmax=800., Nr=401)
dbase.cauchy_rbins_fit(plotrmax=100., plotfig=True, normtype=2)
print dbase.gamma_cauchy_rbins