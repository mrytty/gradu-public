import numpy as np
from scipy.stats import norm
from dcf import dcf


def mini_sigma(a, b, sigma, nu, rho, t, S, T):

    coefficient_a = sigma**2 / (2 * a**3)
    first_term_a = 1 - np.exp(-a * (T - S))
    second_term_a = 1 - np.exp(- 2 * a * (S - t))

    coefficient_b = nu**2 / (2 * b**3)
    first_term_b = 1 - np.exp(-b * (T - S))
    second_term_b = 1 - np.exp(- 2 * b * (S - t))

    coefficient = 2 * rho * sigma * nu / (a * b * (a + b))
    third_term = 1 - np.exp(- (a + b) * (S - t))

    first = coefficient_a * first_term_a**2 * second_term_a
    second = coefficient_b * first_term_b**2 * second_term_b
    third = coefficient * first_term_a * first_term_b * third_term

    return first + second + third


def call_on_zero_coupon_bond(today=None,
                             strike_date=None,
                             maturity=None,
                             strike_price=None,
                             zeros=None,
                             dcf_method=None,
                             **params):

    S = dcf(today, strike_date, method=dcf_method)
    T = dcf(today, maturity, method=dcf_method)

    bond_S = zeros[strike_date]
    bond_T = zeros[maturity]

    a, b = params['k']
    sigma, nu = params['nu']
    rho = params['rho'][0][1]

    sigma_term = np.sqrt(mini_sigma(a, b, sigma, nu, rho, 0, S, T))

    factor = np.log(bond_T / (bond_S * strike_price)) / sigma_term

    return bond_T * norm.cdf(factor + sigma_term / 2) - bond_S * strike_price * norm.cdf(factor - sigma_term / 2)






