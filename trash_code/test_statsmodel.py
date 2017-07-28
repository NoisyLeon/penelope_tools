import statsmodels.api as sm
import numpy as np
nobs = 300
np.random.seed(1234)  # Seed random generator
c1 = np.random.normal(size=(nobs,1))
c2 = np.random.normal(2, 1, size=(nobs,1))
# Estimate a bivariate distribution and display the bandwidth found:

dens_u = sm.nonparametric.KDEMultivariate(data=[c1,c2],
        var_type='cc', bw='normal_reference')