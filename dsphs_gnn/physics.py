
import numpy as np
import scipy
import scipy.special as sc
import scipy.interpolate as interp
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FLRW, Planck18

def gNFW(r, rho_s, r_s, gamma):
    """ Generalized NFW profile """
    x = r / r_s
    return rho_s * x**(-gamma) * (1 + x)**(-3 + gamma)

def beta(r, r_a):
    """ Velocity anisotropy profile Beta(r) """
    return r**2 / (r**2 + r_a**2)

def M_enc(r, rho_s, r_s, gamma):
    """ Enclosed mass profile for a generalized NFW profile. Formula:
    ```
        M_enclosed (r) = 4 pi rho_s r_s^3 \int_0^{r/r_s} x^{2-gamma} (1 + x)^{-3 + gamma} dx
            = 4 pi rho_s r_s^3 (x^{3 - gamma} / (3 - gamma)) 2F1(3-gamma, 3-gamma, 4-gamma, -x)
    ```
        where 2F1 is the hypergeometric function
    Some references:
    - https://dlmf.nist.gov/8.17#E7

    """
    x = r / r_s
    return 4 * np.pi * rho_s * r_s**3 * (
        x**(3-gamma) * sc.hyp2f1(3-gamma, 3-gamma, 4-gamma, -x) / (3-gamma))


def rho_bar(r, rho_s, r_s, gamma):
    """ Mean density profile at each radius """
    return M_enc(r, rho_s, r_s, gamma) / (4 * np.pi * r**3 / 3)


def R_vir(rho_s, r_s, gamma, vir=200, n_steps=10000):
    """ Solve for the virial radius. Default to c200, the radius within which the average
    densityis 200 times the critical density of the Universe at redshift z = 0
    """
    # critical density
    rho_c = Planck18.critical_density(0).to_value(u.Msun / u.kpc**3)

    # calculate enclosed mass and average density
    rho_avg = rho_bar(r, rho_s, r_s, gamma)

    try:
        return interp.interp1d(rho_avg/rho_c, r)(vir)
    except:
        return np.nan


def M_vir(rho_s, r_s, gamma, vir=200, n_steps=10000):
    """ Calculate the virial mass. Default to M200 """
    return M_enc(R_vir(rho_s, r_s, gamma, vir=vir, n_steps=n_steps), rho_s, r_s, gamma)


def c_vir(rho_s, r_s, gamma, vir=200, n_steps=10000):
    """ Calculate the virial mass. Default to c200 """
    return R_vir(rho_s, r_s, gamma, vir=vir, n_steps=n_steps) / r_s


def J_factor(rho_s, r_s, gamma, dc=1, vir=200, n_steps=10000):
    """ Calculate the J factor for a generalized NFW profile. Integrate up to c_vir """
    c = c_vir(rho_s, r_s, gamma, vir, n_steps=n_steps)
    a = 2 * gamma

    J = - np.power(c, 3-a) * np.power(c+1, a-5) * (
        np.power(a, 2) - a*(2*c + 9) + 2*c*(c+5) + 20) / (a-5)*(a-4)*(a-3)
    J = 4 * np.pi * np.power(rho_s, 2) * np.power(r_s, 3) * J / dc**2

    # Convert J from Msun^2 / kpc^5 to GeV^2 / cm^5
    J = J * u.Msun**2 / u.kpc**5
    J = J.to_value(u.GeV**2 / const.c**4 / u.cm**5)
    return J

def poiss_err(n, alpha=0.32):
    """
    Poisson error (variance) for n counts.
    An excellent review of the relevant statistics can be found in
    the PDF statistics review: http://pdg.lbl.gov/2018/reviews/rpp2018-rev-statistics.pdf,
    specifically section 39.4.2.3
    :param: alpha corresponds to central confidence level 1-alpha,
            i.e. alpha = 0.32 corresponds to 68% confidence
    """
    sigma_lo = scipy.stats.chi2.ppf(alpha/2,2*n)/2
    sigma_up = scipy.stats.chi2.ppf(1-alpha/2,2*(n+1))/2
    return sigma_lo, sigma_up


def log10_plummer2d(R, L, r_star):
    """ Log 10 of the Plummer 2D profile
    Args:
        R: projected radius
        params: L, a
    Returns:
        log I(R) = log {L(1 + R^2 / a^2)^{-2} / (pi * r_star^2)}
    """
    logL = np.log10(L)
    logr_star = np.log10(r_star)
    return logL - 2 * logr_star - 2 * np.log10(1 + R**2 / r_star**2) - np.log10(np.pi)


def log10_plummer3d(r, L, r_star):
    """ Log 10 of the Plummer 2D profile
    Args:
        R: projected radius
        params: L, r_star
    Returns:
        log I(R) = log {L(1 + R^2 / a^2)^{-2} / (pi * r_star^2)}
    """
    logL = np.log10(L)
    logr_star = np.log10(r_star)
    return logL - 3 * logr_star - (5/2) * np.log10(1 + r**2 / r_star**2) - np.log10(4 * np.pi / 3)


def calc_nstar(R):
    ''' Calculate the projected number of stars as a function of projected radius R'''
    # calculate the projected radius
    logR = np.log10(R)
    N_star = len(R)

    # binning
    nbins = int(np.ceil(np.sqrt(N_star)))
    logR_min = np.floor(np.min(logR)*10) / 10
    logR_max = np.ceil(np.max(logR)*10) / 10
    n_data, logR_bins = np.histogram(logR, nbins, range=(logR_min, logR_max))

    # ignore bin with zero count
    select = n_data > 0
    n_data = n_data[select]
    logR_bins_lo = logR_bins[:-1][select]
    logR_bins_hi = logR_bins[1:][select]

    # compute poisson error
    n_data_lo, n_data_hi = poiss_err(n_data, alpha=0.32)

    return n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi


def calc_Sigma(R):
    ''' Calculate the projected 2d light profile Sigma(R) where R is the projected radius '''
    n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi =  calc_nstar(R)
    R_bins_lo = 10**logR_bins_lo
    R_bins_hi = 10**logR_bins_hi
    R_bins_ce = 0.5 * (R_bins_lo + R_bins_hi)

    # light profile
    delta_R2 = (R_bins_hi**2 - R_bins_lo**2)
    Sigma_data = n_data / (np.pi * delta_R2)
    Sigma_data_lo = n_data_lo / (np.pi * delta_R2)
    Sigma_data_hi = n_data_hi / (np.pi * delta_R2)

    return Sigma_data, Sigma_data_lo, Sigma_data_hi, R_bins_ce

