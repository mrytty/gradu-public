import numpy as np
from scipy.stats import norm
from dcf import dcf


def blacklet(K, F, vol, omega=1):

    log_ratio = np.log(F / K)


    d1 = (log_ratio + 0.5 * vol**2) / vol
    d2 = (log_ratio - 0.5 * vol**2) / vol

    return F * omega * norm.cdf(omega * d1) - K * omega * norm.cdf(omega * d2)


def caplet_black(bond, forward, S, T, K, sigma, method='Act360'):

    dcf_factor = dcf(S, T, method=method)

    vol = sigma * np.sqrt(S)

    return bond * dcf_factor * blacklet(K, S, forward, vol, omega=1)


def cap_black(bonds, forwards, times, K, sigma, method='Act360'):

    if len(times) == 2:
        return caplet_black(bonds, forwards, times[0], times[1], K, sigma, method=method)

    else:
        sum = 0
        for i in range(len(times) - 1):
            bond = bonds.pop()
            forward = forwards.pop()
            S, T = bond[i], bond[i]

            sum += caplet_black(bond, forward, S, T, K, sigma, method=method)

        return sum


def floorlet_black(bond, forward, S, T, K, sigma, method='Act360'):

    dcf_factor = dcf(S, T, method=method)

    vol = sigma * np.sqrt(S)

    return bond * dcf_factor * blacklet(K, S, forward, vol, omega=-1)


def floor_black(bonds, forwards, times, K, sigma, method='Act360'):
    if len(times) == 2:
        return floorlet_black(bonds, forwards, times[0], times[1], K, sigma, method=method)

    else:

        sum = 0
        for i in range(len(times) - 1):
            bond = bonds.pop()
            forward = forwards.pop()
            S, T = bond[i], bond[i]

            sum += flooret_black(bond, forward, S, T, K, sigma, method=method)

        return sum










