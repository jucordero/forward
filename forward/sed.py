import numpy as np
from astropy.io import fits

def alphas(redshift, alpha_0, alpha_1, z1 = 1):
    """ For a given set of alpha_0, alpha_1 values, return the alpha values for the
    Dirichlet distribution at the given redshift z, evolving from z=0 to z=z1 (z1 = 1 by default)"""

    dirichlet_alphas = np.array([[np.power(a0, 1-(z/z1)) * np.power(a1, (z/z1)) for a0,a1 in zip(alpha_0, alpha_1)] for z in redshift])
    return dirichlet_alphas

def f_lambda(alphas, template_file, ext = 1):
    """ For a given set of alpha, return the c_i weighted SED """
    nz, nt = alphas.shape
    ci = np.zeros_like(alphas)
    for i in range(nz):
        ci[i] = np.random.dirichlet(alphas[i])
    #This assumes SED are on an image file with dimensions (n_lambda, n_template)
    templates = np.array(fits.open(template_file)[ext].data)
    sed = np.array([ci[i]*templates.T for i in range(nz)])
    sed = sed.sum(axis=2)
    return sed
