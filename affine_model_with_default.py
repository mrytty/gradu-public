import numpy as np
import itertools
from scipy.integrate import quad, IntegrationWarning

"""
risk-free: delta + S(a_m X_m) + S(c_i Y_i) 
spread: spread + S(b_m X_m) + S(d_i Y_i)
"""


def beta_m(alpha_m, sigma_m, a_m, b_m):
    """
    Implements Eq. from Nawalka, Beliaeva, Soto (pg. 460)
    """
    if b_m:
        changing = np.array(a_m) + np.array(b_m)
    else:
        changing = np.array(a_m)

    return np.sqrt(alpha_m ** 2 + 2 * changing * sigma_m ** 2)


def C_i(tau, **params):
    """
    Implements  Eq. 9.126 from Nawalka, Beliaeva, Soto (pg. 460)
    Need params: k
    """
    numerator = 1 - np.exp(- params['k'] * tau)
    return numerator / params['k']


def B_m(tau, **params):
    """
    Implements  Eq. 9.125 from Nawalka, Beliaeva, Soto (pg. 460)
    Needs params: alpha, sigma
    """
    # Get beta
    if params['risk_free']:
        beta = beta_m(params['alpha'], params['sigma'], params['a_m'], 0)

    else:
        beta = beta_m(params['alpha'], params['sigma'], params['a_m'], params['b_m'])

    # Terms
    exp_term = np.exp(beta * tau) - 1
    denom = (beta + params['alpha']) * exp_term + 2 * beta

    return 2 * exp_term / denom


def first_part_A(tau, **params):
    """
    Implements first part of Eq. 9.124 from Nawalka, Beliaeva, Soto (pg. 460)
    Needs params: alpha, beta, theta, sigma
    """
    # Get beta
    if params['risk_free']:
        beta = beta_m(params['alpha'], params['sigma'], params['a_m'], 0)

    else:
        beta = beta_m(params['alpha'], params['sigma'], params['a_m'], params['b_m'])


    # Terms
    log_term_nom = 2 * beta * np.exp((beta + params['alpha']) * tau / 2)
    log_term_denom = (beta + params['alpha']) * (np.exp(beta * tau) - 1) + 2 * beta
    coef = 2 * params['alpha'] * params['theta'] / params['sigma'] ** 2

    return np.sum(coef * np.log(log_term_nom / log_term_denom))


def last_part_A(tau, **params):
    """
    Implements last part of Eq. 9.124 from Nawalka, Beliaeva, Soto (pg. 460)
    Needs params: k, nu, rho
    """

    # Get C
    C = C_i(tau, **params)

    l = params['N'] - params['M']

    if params['risk_free']:
        changing = np.array(params['c_i'])

    else:
        changing = np.array(params['c_i']) + np.array(params['d_i'])

    # Summation
    part_sum = 0
    for i, j in itertools.product(range(l), range(l)):
        if l == 1:
            rho = 1
        else:
            rho = params['rho'][i][j]

        new_term = changing[i] * changing[j]
        coef = new_term * params['nu'][i] * params['nu'][j] / (params['k'][i] * params['k'][j]) * rho
        term = (1 - np.exp(- (params['k'][i] + params['k'][j]) * tau)) / (params['k'][i] + params['k'][j])
        part_sum += coef * (tau - C[i] - C[j] + term)

    return part_sum / 2


def A_fun(tau, **params):
    gaussian, non_gaussian = 0, 0

    if params['M'] > 0:
        non_gaussian = first_part_A(tau, **params)

    if params['N'] - params['M'] > 0:
        gaussian = last_part_A(tau, **params)

    return gaussian + non_gaussian


def H_simple(t, T, **params):
    """
    Implements Eq. 9.123 from Nawalka, Beliaeva, Soto (pg. 460)
    """
    if params['risk_free']:
        delta = params['delta']

    else:
        delta = params['delta'] + params['spread']

    return delta * (T - t)


def bond_pricer_simple(t, T, **params):
    """
    Implements Eq. 9.31 from Nawalka, Beliaeva, Soto (pg. 426)
    """
    tau = T - t
    A = A_fun(tau, **params)
    B_term, C_term = 0, 0

    if params['M'] > 0:

        if params['risk_free']:
            new_term = np.array(params['a_m'])
        else:
            new_term = np.array(params['a_m']) + np.array(params['b_m'])

        B_term = np.sum(new_term * B_m(tau, **params) * params['X_0'])

    if params['N'] - params['M'] > 0:

        if params['risk_free']:
            new_term = np.array(params['c_i'])
        else:
            new_term = np.array(params['c_i']) + np.array(params['d_i'])

        C_term = np.sum(new_term * C_i(tau, **params) * params['Y_0'])

    H = H_simple(t, T, **params)

    return np.exp(A - B_term - C_term - H)


def beta1m(**params):
    """
    Implements Eq. 9.162 from Nawalka, Beliaeva, Soto (pg. 472)
    """
    inside = params['alpha']**2 + 2 * (np.array(params['a_m']) + np.array(params['b_m'])) * params['sigma']**2
    return np.sqrt(inside)


def beta2m(beta1, **params):
    """
    Implements Eq. 9.162 from Nawalka, Beliaeva, Soto (pg. 472)
    """

    return (beta1 - params['alpha']) / 2


def beta3m(beta1, **params):
    """
    Implements Eq. 9.162 from Nawalka, Beliaeva, Soto (pg. 472)
    """

    return (- beta1 - params['alpha']) / 2


def beta4m(phi, beta1, **params):
    """
    Implements Eq. 9.162 from Nawalka, Beliaeva, Soto (pg. 472)
    """

    last_term = phi * np.array(params['b_m']) * params['sigma']**2
    numerator = - params['alpha'] - beta1 + last_term
    denominator = - params['alpha'] + beta1 + last_term

    return numerator / denominator


def q_i(phi, **params):
    """
    Implements Eq. 9.161 from Nawalka, Beliaeva, Soto (pg. 472)
    """
    fraction = np.array(params['d_i']) / (np.array(params['c_i']) + np.array(params['d_i']))

    return 1 + phi * params['k'] * fraction


def C_i_def(tau, phi, **params):
    """
    Implements Eq. 9.160 from Nawalka, Beliaeva, Soto (pg. 472)
    """
    q = q_i(phi, **params)
    return (1 - q * np.exp(- params['k'] * tau)) / params['k']


def B_m_def(tau, phi, **params):
    """
    Implements Eq. 9.159 from Nawalka, Beliaeva, Soto (pg. 471)
    """

    beta1 = beta1m(**params)
    beta2 = beta2m(beta1, **params)
    beta3 = beta3m(beta1, **params)
    beta4 = beta4m(phi, beta1, **params)
    exp_term = np.exp(beta1 * tau)
    denominator = beta2 * beta4 * exp_term - beta3
    numerator = beta4 * exp_term - 1
    term = np.array(params['a_m']) + np.array(params['b_m'])

    return 2 / (term * params['sigma']**2) * denominator / numerator


def A_first_sum_def(tau, phi, **params):
    """
    Implements Eq. 9.158 from Nawalka, Beliaeva, Soto (pg. 471)
    """

    summa = 0

    l = params['N'] - params['M']

    for i, j in itertools.product(range(l), range(l)):

        if l == 1:
            rho = 1
        else:
            rho = params['rho'][i][j]

        cplusd = np.array(params['c_i']) + np.array(params['d_i'])

        q = q_i(phi, **params)
        C = C_i_def(tau, phi, **params)

        factor = cplusd[i] * cplusd[j] * params['nu'][i] * params['nu'][j] * rho / params['k'][i] / params['k'][j]

        k_sum = params['k'][i] + params['k'][j]

        fraction = (1 - np.exp(- k_sum * tau) / k_sum)
        term = tau - q[i] * C[i] - q[j] * C[j] + q[i] * q[j] * fraction

        summa += factor * term

    return summa / 2


def A_second_sum_def(tau, phi, **params):
    """
    Implements Eq. 9.158 from Nawalka, Beliaeva, Soto (pg. 471)
    """

    beta1 = beta1m(**params)
    beta3 = beta3m(beta1, **params)
    beta4 = beta4m(phi, beta1, **params)

    fraction = params['alpha'] * params['theta'] / params['sigma']**2

    ln_denominator = 1 - beta4 * np.exp(beta1 * tau)
    ln_numerator = 1 - beta4

    return 2 * sum(fraction * (beta3 * tau + np.log(ln_denominator / ln_numerator)))


def A_sum_def(tau, phi, **params):
    """
    Implements Eq. 9.157 from Nawalka, Beliaeva, Soto (pg. 471)
    """

    A_first, A_last = 0, 0

    if params['M'] > 0:
        A_last = A_second_sum_def(tau, phi, **params)

    if params['N'] - params['M'] > 0:
        A_first = A_first_sum_def(tau, phi, **params)

    return phi * params['spread'] + A_first - A_last


def H_def(t, T, **params):

    return (T - t) * (params['delta'] + params['spread'])


def eta(t, T, phi, **params):

    tau = T - t

    if params['M'] > 0:
        factor = np.array(params['a_m']) + np.array(params['b_m'])
        B_term = sum(factor * B_m_def(tau, phi, **params) * params['X_0'])

    else:
        B_term = 0

    if params['N'] - params['M'] > 0:
        factor = np.array(params['c_i']) + np.array(params['d_i'])
        C_term = sum(factor * C_i_def(tau, phi, **params) * params['Y_0'])

    else:
        C_term = 0

    A_term = A_sum_def(tau, phi, **params)

    H = H_def(t, T, **params)

    return np.exp(A_term - B_term - C_term - H)


def G_function(t, T, h, **params):

    return (eta(t, T, h, **params) - eta(t, T, 0, **params)) / h


def in_default(t, T, **params):

    tau = T - t
    h = 10**-5

    def integrand(x):

        return G_function(t, t + x, h, **params)

    try:
        with_recovery = quad(integrand, 0, tau, full_output=1)

    except IntegrationWarning:
        with_recovery = [0, 0, 0]

    else:
        with_recovery = [0, 0, 0]

    if 'LGD' in params:
        return with_recovery[0] * (1 - params['LGD'])

    else:
        return with_recovery[0]


def defaultable_bond_pricer_simple_with_recovery(t, T, **params):
    """
    Implements Eq. 9.149 from Nawalka, Beliaeva, Soto (pg. 470)
    """

    tau = T - t

    without_recovery = bond_pricer_simple(t, T, **params)

    if 'risk_free' in params:
        if params['risk_free']:
            return without_recovery

    if 'LGD' in params:
        if params['LGD'] is None:
            return without_recovery
        elif params['LGD'] == 1:
            return without_recovery

    with_recovery = in_default(t, T, **params)

    return without_recovery + with_recovery



def CDS_premium_coupon(t, T, cds_spread, **params):

    return cds_spread * bond_pricer_simple(t, T, **params)


def CDS_protection_leg(t, T, **params):

    return in_default(t, T, **params)


def CDS_continous_price(t, T, cds_spread, **params):

    tau = T - t
    protection_leg = CDS_protection_leg(t, T, **params)

    def integrand(x):
        return bond_pricer_simple(t, t + x, **params)

    integration = quad(integrand, 0, tau)
    premium_leg = integration[0] * cds_spread

    return premium_leg - protection_leg




"""
def_test_01 = dict(M=0, N=1, delta=0.02,
                   c_i=[1], k=np.array([1]), nu=np.array([0.02]), Y_0=0.03,
                   risk_free=True)
test_12 = dict(M=1, N=2, delta=0.02, alpha=np.array([2]), theta=np.array([0.04]), sigma=np.array([0.025]), X_0=0.06,
                   k=np.array([1]), nu=np.array([0.02]), Y_0=0.03,
                   risk_free=True)
def_test_12 = dict(M=1, N=2, delta=0.02, spread=0.01,
                   a_m=[1], b_m=[1], alpha=np.array([2]), theta=np.array([0.04]), sigma=np.array([0.025]), X_0=0.06,
                   c_i=[1], d_i=[1], k=np.array([1]), nu=np.array([0.02]), Y_0=0.03,
                   risk_free=False, LGD=0.4)
bond_pricer_simple(0, 1, **def_test_12)

G_function(0, 1, 10**-6, **def_test_12)

from affine_model import bond_pricer_simple as test_function
test_function(0, 1, **test_12)

def_test_13 = dict(M=1, N=3, delta=0.02, spread=0.01,
                   a_m=[0], b_m=[1], alpha=np.array([2]), theta=np.array([0.04]), sigma=np.array([0.025]), X_0=0.06,
                   c_i=[1, 1], d_i=[1, 0], k=np.array([1, 1.5]), nu=np.array([0.02, 0.015]), Y_0=[0.03, 0], rho=[[1, -0.5], [-0.5, 1]],
                   risk_free=False, LGD=0.4)
"""

