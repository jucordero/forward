import numpy as np
from astropy.io import fits

def alphas(z, alpha_0, alpha_1, z1 = 1):
    """ For a given set of alpha_0, alpha_1 values, return the alpha values for the
    Dirichlet distribution at the given redshift z, evolving from z=0 to z=z1 (z1 = 1 by default)"""

    return alpha = np.power(alpha_0, 1-(z/z1)) * np.power(alpha_1, (z/z1))

def f_lambda(alphas, template_file, ext = 1):
    """ For a given set of alpha, return the c_i weighted SED """
    ci = np.random.dirichlet(alphas)
    #This assumes SED are on an image file with dimensions (n_lambda, n_template)
    templates = np.array(fits.open(template_file)[ext].data)
    sed = ci*templates.sum(axis = 1)
    return sed
