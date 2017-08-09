import database

pfx     = 'I_10nA_E_30keV'
infname = 'from_Amrita/Au_100nm/'+pfx+'/psf-test.dat'
dbase   = database.penelopedbase()
dbase.read_psf(infname)

dbase.count_r_bins( rmax=800., Nr=401)
# dbase.gauss_rbins_fit(plotrmax=400., plotfig=True, normtype=2)
# dbase.count_r_bins(zmin=90., rmax=800., Nr=401)

dbase.gauss_rbins_fit(plotrmax=200., plotfig=True, normtype=2)
# dbase.cauchy_rbins_fit(plotrmax=400., plotfig=True, normtype=0.5)

dbase.pdf_radius(ratio=0.9)
# print dbase.gamma_cauchy_rbins