import numpy as np
from scipy.optimize import differential_evolution, minimize
from differential_evolution import DifferentialEvolution
from near_corr_matrix import nearcorr, ExceededMaxIterationsError

import time


"""
Affine parameters explained.
A(M,N) -model with n = N-M gaussian and M square-root processes.
A square-root process has 4 parameters:
    alfa (speed of mean reversion),
    theta (mean value),
    sigma (diffusion)
    X_0 (initial state)
A gaussian process has 2 free parameters (if with dynamic expansion):
    k (speed of mean reversion)
    nu (diffusion)
    and Y_0 (initial state) is chosen to be 0
A gaussian process has 3 free parameters (if no dynamic expansion):
    k (speed of mean reversion)
    nu (diffusion)
    and Y_0 (initial state) is to be calibrated
n gaussian process also has n(n-1)/2 parameters for correlations

A(M,N)+ also has the shift parameter.

A(M,N)+ has 
1 + 3 * n  +  4 * M  +  1_(n>0) ( n(n-1)/2 )
free parameters to be calibrated.

A(M,N)++ has
2 * n  +  4 * M  +  1_(n>0) ( n(n-1)/2 )
free parameters to be calibrated.
"""


def generate_bounds_for_gaussian_process(n,
                                         simple=True,
                                         k=(0.001, 3),
                                         nu=(0.001, 0.25),
                                         rho=(-0.99, 0.99),
                                         y0=(-0.1, 0.1)):
    bounds = []

    for i in range(n):
        # k, speed of mean reversion
        bounds.append(k)

    for i in range(n):
        # nu, diffusion
        bounds.append(nu)

    if n > 1:
        k = int(n * (n - 1) / 2)
        for i in range(k):
            bounds.append(rho)

    # For simple process, we need initial state values.
    # For extended, we assume them to be zeros, so we need no bounds
    if simple:

        for i in range(n):
            # Y_0, initial state
            bounds.append(y0)

    return bounds


def generate_bounds_for_square_root_process(M,
                                            alpha=(0.001, 3),
                                            theta=(0, 0.1),
                                            sigma=(0.001, 0.25),
                                            x0=(0.001, 0.1)):

    bounds = []

    for i in range(M):
        # alpha, speed of mean reversion
        bounds.append(alpha)

    for i in range(M):
        # theta, mean value
        bounds.append(theta)

    for i in range(M):
        # sigma, diffusion
        bounds.append(sigma)

    for i in range(M):
        # X_0, initial state
        bounds.append(x0)

    return bounds


def generate_bounds(M,
                    N,
                    simple=True,
                    shift=(-0.1, 0.1),
                    alpha=(0.001, 3),
                    theta=(0, 0.1),
                    sigma=(0.001, 0.25),
                    x0=(0.001, 0.1),
                    k=(0.001, 3),
                    nu=(0.001, 0.25),
                    rho=(-0.99, 0.99),
                    y0=(-0.1, 0.1)):

    bounds = []

    if shift:
        if simple:
            bounds.append(shift)

    if M > 0:
        bounds += generate_bounds_for_square_root_process(M,
                                                          alpha=alpha,
                                                          theta=theta,
                                                          sigma=sigma,
                                                          x0=x0)

    if N - M > 0:
        bounds += generate_bounds_for_gaussian_process(N - M,
                                                       simple=simple,
                                                       k=k,
                                                       nu=nu,
                                                       rho=rho,
                                                       y0=y0)

    return bounds


def square_root_array_to_param(array, M=0):

    d = {}

    if M > 0:

        d['alpha'], array = array[:M], array[M:]
        d['theta'], array = array[:M], array[M:]
        d['sigma'], array = array[:M], array[M:]
        d['X_0'], array = array[:M], array[M:]

    return d


def gaussian_array_to_param(array, n=1, simple=True):

    d = {}

    if n > 0:

        d['k'], array = array[:n], array[n:]
        d['nu'], array = array[:n], array[n:]

        if n > 1:

            # Initilize correlation matrix
            rho = np.eye(n)
            # Get the positions of upper-triangle above the diagonal
            upper_indices = np.triu_indices(n, 1)

            # Write the correlations
            for i, j in zip(upper_indices[0], upper_indices[1]):
                # Pop the value
                corr, array = array[0], array[1:]
                rho[i, j] = corr
                # Symmetric
                rho[j, i] = corr

            try:
                d['rho'] = nearcorr(rho)

            except ExceededMaxIterationsError:
                d['rho'] = rho


        if simple:
            d['Y_0'], array = array[:n], array[n:]

        # If extended model, we assume that initial states are zeros.
        else:
            d['Y_0'] = np.zeros(n)

    return d


def array_to_param(array, M=0, N=0, simple=True):

    n = N - M

    d = dict(delta=0,
             M=M,
             N=N)

    if simple:
        d['shift'], array = array[0], array[1:]

    else:
        d['shift'] = 0

    # Params for square-root processes.
    k = 4 * M

    square_root_params, gaussian_params = array[:k], array[k:]
    if M > 0:
        d.update(square_root_array_to_param(square_root_params, M=M))

    if n > 0:
        d.update(gaussian_array_to_param(gaussian_params, n=n, simple=simple))

    return d


def define_optimization_function(diff_fun,
                                 M=0,
                                 N=0,
                                 simple=True,
                                 instruments=None,
                                 pricer=None,
                                 scale=1000000):

    def fun(param_array):

        if simple:
            params = array_to_param(param_array,
                                    M=M,
                                    N=N,
                                    simple=simple)
        else:
            params = array_to_param(param_array,
                                    M=M,
                                    N=N,
                                    simple=simple)

        d_sum = 0

        for deriv in instruments:
            d_sum += scale * diff_fun(deriv.calibrate, pricer.price(deriv, **params))

        return d_sum

    return fun


def squared(x, y):

    return (x - y)**2


def relative_squared(x, y):

    return ((x - y) / x)**2


def calibration(M=0, N=0, simple=True, difference_function=squared, instruments=None, pricer=None):

    bounds = generate_bounds(M=M, N=N, simple=simple)

    optimization_fun = define_optimization_function(difference_function,
                                                    M=M,
                                                    N=N,
                                                    simple=simple,
                                                    instruments=instruments,
                                                    pricer=pricer,
                                                    scale=100)

    start = time.time()
    res = differential_evolution(optimization_fun,
                                 bounds,
                                 strategy='best1bin',
                                 maxiter=2500,
                                 popsize=30,
                                 tol=0.01,
                                 mutation=(0.5, 1.2),
                                 recombination=0.7,
                                 polish=True)
    end = time.time()
    duration = np.round(end - start, 2)
    print('Calibration took {} seconds.'.format(duration))
    print(res)

    return res


def own_calibration(M=0,
                    N=0,
                    gen_seed=None,
                    opt_seed=None,
                    simple=True,
                    difference_function=squared,
                    instruments=None,
                    pricer=None,
                    bounds=None,
                    tol=0.001,
                    generations=1000,
                    size=1024,
                    scale=100000000,
                    min_population=32,
                    culling=(0.2, 0.6),
                    generational_cycle=10,
                    return_function=False):

    if bounds is None:
        bounds = dict(shift=(-0.1, 0.1),
                      alpha=(0.001, 3),
                      theta=(0, 0.1),
                      sigma=(0.001, 0.25),
                      x0=(0.001, 0.1),
                      k=(0.001, 3),
                      nu=(0.001, 0.25),
                      rho=(-0.99, 0.99),
                      y0=(-0.1, 0.1))


    calibration_object = DifferentialEvolution()

    bounds = generate_bounds(M=M, N=N, simple=simple, **bounds)

    calibration_object.set_bounds(bounds)
    calibration_object.generate_initial_population(size=size, seed=gen_seed)

    optimization_fun = define_optimization_function(difference_function,
                                                    M=M,
                                                    N=N,
                                                    simple=simple,
                                                    instruments=instruments,
                                                    pricer=pricer,
                                                    scale=scale)

    start = time.time()
    res_evo = calibration_object.optimize(func=optimization_fun,
                                          generations=generations,
                                          tol=tol,
                                          min_population=min_population,
                                          culling=culling,
                                          generational_cycle=generational_cycle,
                                          seed=opt_seed
                                          )

    res = minimize(optimization_fun,
                   x0=res_evo.x[0],
                   bounds=bounds,
                   method='L-BFGS-B')

    end = time.time()
    duration = np.round(end - start, 2)
    print('Calibration took {} seconds.'.format(duration))
    print(res)

    if return_function:
        return res, res_evo, calibration_object, duration, optimization_fun

    return res, res_evo, calibration_object, duration


"""
res = calibration(M=0,
                  N=1,
                  simple=True,
                  instruments=c.generate_zeros(dates=test_dates),
                  pricer=simpler)
"""





















