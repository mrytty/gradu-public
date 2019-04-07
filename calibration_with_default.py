import numpy as np
import copy
from scipy.optimize import differential_evolution, minimize
from differential_evolution import DifferentialEvolution
from near_corr_matrix import nearcorr, ExceededMaxIterationsError

import time


"""
Affine parameters explained. With DEFAULT
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

Default also adds 2 new parameters per gaussian process (c_i, d_i)
and 2 new parameters per square-root process, but we set all of them to be binary and they are set
before optimization. So these are not to be more parameters to be optimized

A(M,N)+ also has the shift parameter.

A(M,N)+ has 
1 + 3 * n  +  4 * M  +  1_(n>0) ( n(n-1)/2 )
free parameters to be calibrated.

A(M,N)++ has
2 * n  +  4 * M  +  1_(n>0) ( n(n-1)/2 )
free parameters to be calibrated.

The ordering is  for A(M,N)+ models
1 delta
1 spread
4*M square-root parameters (X_0, alpha, theta, sigma)
3*n gaussian parameters(Y_0, k, nu)

risk-free: delta + S(a_m X_m) + S(c_i Y_i) 
spread: spread + S(b_m X_m) + S(d_i Y_i)
where S is sum

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
                    delta=(-0.1, 0.1),
                    spread=(0, 0.1),
                    alpha=(0.001, 3),
                    theta=(0, 0.1),
                    sigma=(0.001, 0.25),
                    x0=(0.001, 0.1),
                    k=(0.001, 3),
                    nu=(0.001, 0.25),
                    rho=(-0.99, 0.99),
                    y0=(-0.1, 0.1),
                    LGD=(0, 1)):

    bounds = []

    if delta:
        bounds.append(delta)

    if spread:
        bounds.append(spread)

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

    if LGD is not None:
        bounds.append(LGD)

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

    d = dict(M=M,
             N=N,
             shift=0)

    # First delta and then delta_spread
    d['delta'], array = array[0], array[1:]
    d['spread'], array = array[0], array[1:]

    # Params for square-root processes.
    l = 4 * M

    square_root_params, gaussian_params = array[:l], array[l:]
    if M > 0:
        d.update(square_root_array_to_param(square_root_params, M=M))

    if n > 0:
        d.update(gaussian_array_to_param(gaussian_params, n=n, simple=simple))

    return d


class ParamHandler:

    def __init__(self, M=0, N=0, a_m=None, b_m=None, c_i=None, d_i=None, LGD=None, simple=True):

        self._M = M
        self._N = N
        self._n = N - M

        self._simple = simple

        self._LGD = LGD

        if a_m is None:
            self._a_m = [1] * M
        elif len(a_m) == M:
            self._a_m = a_m
        else:
            raise ValueError

        if b_m is None:
            self._b_m = [1] * M
        elif len(b_m) == M:
            self._b_m = b_m
        else:
            raise ValueError

        if c_i is None:
            self._c_i = [1] * self._n
        elif len(c_i) == self._n:
            self._c_i = c_i
        else:
            raise ValueError

        if d_i is None:
            self._d_i = [1] * self._n
        elif len(d_i) == self._n:
            self._d_i = d_i
        else:
            raise ValueError

    def generate_param_dicts(self, array):

        # Add parameters from optimizer array
        riskfree_dict = array_to_param(array, M=self._M, N=self._N, simple=self._simple)

        # Add fixed coefficients
        riskfree_dict['a_m'] = self._a_m
        riskfree_dict['b_m'] = self._b_m
        riskfree_dict['c_i'] = self._c_i
        riskfree_dict['d_i'] = self._d_i

        # LGD information, if self._LGD is None, a value is not given and therefore it is optimized also
        # It is the last value in the array
        if self._LGD is None:
            riskfree_dict['LGD'] = array[-1]

        # Otherwise give the hard-coded value
        else:
            riskfree_dict['LGD'] = self._LGD

        # Make dict for defaultable instrument
        def_dict = copy.copy(riskfree_dict)

        # Is risk-free
        riskfree_dict['risk_free'] = True

        # Is defaultable
        def_dict['risk_free'] = False

        return riskfree_dict, def_dict

    def generate_bounds(self,
                        delta=(-0.1, 0.1),
                        spread=(0, 0.1),
                        alpha=(0.001, 3),
                        theta=(0, 0.1),
                        sigma=(0.001, 0.25),
                        x0=(0.001, 0.1),
                        k=(0.001, 3),
                        nu=(0.001, 0.25),
                        rho=(-0.99, 0.99),
                        y0=(-0.1, 0.1),
                        LGD=(0,1)):

        return generate_bounds(M=self._M,
                               N=self._N,
                               simple=True,
                               delta=delta,
                               spread=spread,
                               alpha=alpha,
                               theta=theta,
                               sigma=sigma,
                               x0=x0,
                               k=k,
                               nu=nu,
                               rho=rho,
                               y0=y0,
                               LGD=LGD)


class ParamHandlerWithFixed:

    def __init__(self,
                 M=0,
                 N=0,
                 fixed_param=None,
                 a_m=None,
                 b_m=None,
                 c_i=None,
                 d_i=None,
                 LGD=None,
                 simple=True):
        """
        :param fixed_param:
        :param a_m:
        :param b_m:
        :param c_i:
        :param d_i:
        :param LGD:
        :param simple:
        """

        self._fixed_param = fixed_param

        self._fixedM = fixed_param['M']
        self._fixedN = fixed_param['N']
        self._fixedn = self._fixedN - self._fixedM
        if self._fixedM <= M:
            self._M = M
        else:
            raise ValueError

        if self._fixedN <= N:
            self._N = N
        else:
            raise ValueError

        self._n = N - M

        self._simple = simple

        if LGD is not None:
            self._LGD = LGD

        else:
            if 'LGD' in fixed_param:
                self._LGD = fixed_param['LGD']
            else:
                self._LGD = None

        if a_m is None:
            self._a_m = [1] * M
        elif len(a_m) == M:
            self._a_m = a_m
        else:
            raise ValueError

        if b_m is None:
            self._b_m = [1] * M
        elif len(b_m) == M:
            self._b_m = b_m
        else:
            raise ValueError

        if c_i is None:
            self._c_i = [1] * self._n
        elif len(c_i) == self._n:
            self._c_i = c_i
        else:
            raise ValueError

        if d_i is None:
            self._d_i = [1] * self._n
        elif len(d_i) == self._n:
            self._d_i = d_i
        else:
            raise ValueError

    def generate_bounds_for_gaussian_process(self,
                                             k=(0.001, 3),
                                             nu=(0.001, 0.25),
                                             rho=(-0.99, 0.99),
                                             y0=(-0.1, 0.1)
                                             ):

        l = self._n - self._fixedn

        bounds = []

        for i in range(l):
            # k, speed of mean reversion
            bounds.append(k)

        for i in range(l):
            # nu, diffusion
            bounds.append(nu)

        no_of_new_rhos = int((self._n * (self._n - 1)) / 2 - (self._fixedn * (self._fixedn - 1)) / 2)

        for i in range(no_of_new_rhos):
            bounds.append(rho)

        for i in range(l):
            # Y_0, initial state
            bounds.append(y0)

        return bounds

    def generate_bounds(self,
                        spread=(0, 0.15),
                        alpha=(0.001, 3),
                        theta=(0, 0.1),
                        sigma=(0.001, 0.25),
                        x0=(0.001, 0.1),
                        k=(0.001, 3),
                        nu=(0.001, 0.25),
                        rho=(-0.99, 0.99),
                        y0=(-0.1, 0.1),
                        LGD=(0, 1)):

        bounds = []

        if spread:
            bounds.append(spread)

        if self._M - self._fixedM > 0:
            bounds += generate_bounds_for_square_root_process(M=self._M - self._fixedM,
                                                              alpha=alpha,
                                                              theta=theta,
                                                              sigma=sigma,
                                                              x0=x0)

        if self._n - self._fixedn > 0:
            bounds += self.generate_bounds_for_gaussian_process(k=k,
                                                                nu=nu,
                                                                rho=rho,
                                                                y0=y0)

        if self._LGD is not None:
            if LGD is not None:
                bounds.append(LGD)

        return bounds

    @staticmethod
    def num_handler(num):

        if hasattr(num, 'len'):
            return np.array([num])

        else:
            return np.array(num)

    def generate_param_dicts(self, array):

        param = copy.copy(self._fixed_param)
        param['a_m'] = self._a_m
        param['b_m'] = self._b_m
        param['c_i'] = self._c_i
        param['d_i'] = self._d_i

        param['spread'], array = array[0], array[1:]

        M_diff = self._M - self._fixedM
        if M_diff > 0:

            # If we have no fixed square-root process, then we may just generate dict as usually
            if self._fixedM == 0:

                # Params for square-root processes.
                l = 4 * M_diff

                square_root_params, array = array[:l], array[l:]
                param.update(square_root_array_to_param(square_root_params, M=M_diff))

            # Otherwise we have to append
            else:
                alpha, array = array[:M_diff], array[M_diff:]
                theta, array = array[:M_diff], array[M_diff:]
                sigma, array = array[:M_diff], array[M_diff:]
                X_0, array = array[:M_diff], array[M_diff:]

                param['alpha'] = np.append(self.num_handler(self._fixed_param['alpha']), self.num_handler(alpha))
                param['theta'] = np.append(self.num_handler(self._fixed_param['theta']), self.num_handler(theta))
                param['sigma'] = np.append(self.num_handler(self._fixed_param['sigma']), self.num_handler(sigma))
                param['X_0'] = np.append(self.num_handler(self._fixed_param['X_0']), self.num_handler(X_0))

        n_diff = self._n - self._fixedn
        if n_diff > 0:

            # If no fixed gaussian process, then we may just generate gaussian dict as usually
            if self._fixedn == 0:

                # Params for gaussian processes.
                l = int(3 * n_diff + n_diff * (n_diff - 1) / 2)

                gaussian_params, array = array[:l], array[l:]
                param.update(gaussian_array_to_param(gaussian_params, n=n_diff, simple=True))

            # Thus we have at least one fixed gaussian process and at least one new gaussian process
            else:

                k, array = array[:n_diff], array[n_diff:]
                nu, array = array[:n_diff:], array[n_diff:]

                # this is the number of new correlation parameters
                l = int((self._n * (self._n - 1)) / 2 - (self._fixedn * (self._fixedn - 1)) / 2)
                array_for_rho, array = array[:l], array[l:]

                Y_0, array = array[:n_diff], array[n_diff:]

                param['k'] = np.append(self.num_handler(self._fixed_param['k']), self.num_handler(k))
                param['nu'] = np.append(self.num_handler(self._fixed_param['nu']), self.num_handler(nu))
                param['Y_0'] = np.append(self.num_handler(self._fixed_param['Y_0']), self.num_handler(Y_0))

                # Now we generate correlation matrix
                rho = np.eye(self._n)

                xs, ys = np.triu_indices(self._n, 1)
                whole_triangle_idx = zip(xs, ys)

                # If there is only one fixed gaussian process, then there are no previous correlation parameters
                if self._fixedn == 1:

                    for i, j in whole_triangle_idx:
                        rho[i, j], array_for_rho = array_for_rho[0], array_for_rho[1:]
                        rho[j, i] = rho[i, j]

                # Otherwise there is a fixed correlation matrix of size
                # (self._fixedn x self._fixedn), self._fixedn > 1
                else:

                    xxs, yys = np.triu_indices(self._fixedn, 1)
                    upper_triangle_idx = zip(xxs, yys)

                    # Rho upper indices
                    for i, j in whole_triangle_idx:

                        # If it is in the fixed, read from fixed params
                        if (i, j) in upper_triangle_idx:
                            rho[i, j] = self._fixed_param['rho'][i, j]
                            rho[j, i] = self._fixed_param['rho'][i, j]

                        # Otherwise it is from the array
                        else:
                            rho[i, j], array_for_rho = array_for_rho[0], array_for_rho[1:]
                            rho[j, i] = rho[i, j]

                try:
                    param['rho'] = nearcorr(rho)

                except ExceededMaxIterationsError:
                    param['rho'] = rho

        if self._LGD is None:
            param['LGD'] = array[-1]

        # Otherwise give the hard-coded value
        else:
            param['LGD'] = self._LGD

        # Make dict for defaultable instrument
        def_dict = copy.copy(param)

        # Is risk-free
        param['risk_free'] = True

        # Is defaultable
        def_dict['risk_free'] = False

        return param, def_dict


def define_optimization_function(diff_fun,
                                 M=0,
                                 N=0,
                                 simple=True,
                                 riskless_instruments=None,
                                 risky_instruments=None,
                                 a_m=None,
                                 b_m=None,
                                 c_i=None,
                                 d_i=None,
                                 LGD=0.4,
                                 pricer=None,
                                 fixed_param=None,
                                 scale=1000000):

    if fixed_param is None:
        handler = ParamHandler(M=M,
                               N=N,
                               a_m=a_m,
                               b_m=b_m,
                               c_i=c_i,
                               d_i=d_i,
                               LGD=LGD,
                               simple=simple)

    else:
        handler = ParamHandlerWithFixed(M=M,
                                        N=N,
                                        a_m=a_m,
                                        b_m=b_m,
                                        c_i=c_i,
                                        d_i=d_i,
                                        LGD=LGD,
                                        fixed_param=fixed_param,
                                        simple=True)

    def fun(param_array):

        riskfree_dict, risky_dict = handler.generate_param_dicts(param_array)

        d_sum = 0

        if fixed_param is None:
            for deriv in riskless_instruments:
                d_sum += scale * diff_fun(deriv.calibrate, pricer.price(deriv, **riskfree_dict))

        for deriv in risky_instruments:
            d_sum += scale * diff_fun(deriv.calibrate, pricer.price(deriv, **risky_dict))

        return d_sum

    return fun, handler


def squared(x, y):
    return (x - y) ** 2


def relative_squared(x, y):
    if x == 0:
        relative = (x - y) / 10 ** -8

    else:
        relative = (x - y) / x

    return relative ** 2


def calibration(M=0,
                N=0,
                simple=True,
                difference_function=squared,
                riskless_instruments=None,
                risky_instruments=None,
                a_m=None,
                b_m=None,
                c_i=None,
                d_i=None,
                LGD=0.4,
                fixed_param=None,
                pricer=None):

    optimization_fun, handler = define_optimization_function(difference_function,
                                                             M=M,
                                                             N=N,
                                                             simple=simple,
                                                             riskless_instruments=riskless_instruments,
                                                             risky_instruments=risky_instruments,
                                                             a_m=a_m,
                                                             b_m=b_m,
                                                             c_i=c_i,
                                                             d_i=d_i,
                                                             LGD=LGD,
                                                             pricer=pricer,
                                                             fixed_param=fixed_param,
                                                             scale=10000)

    bounds = handler.generate_bounds()

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
                    tol=0.001,
                    generations=100,
                    size=256,
                    scale=100000000,
                    min_population=32,
                    culling=(0.2, 0.6),
                    bounds=None,
                    generational_cycle=5,
                    riskless_instruments=None,
                    risky_instruments=None,
                    a_m=None,
                    b_m=None,
                    c_i=None,
                    d_i=None,
                    LGD=None,
                    final_LGD=0.4,
                    fixed_param=None,
                    pricer=None,
                    return_handler=False):

    # DE optimization
    optimization_fun, handler = define_optimization_function(difference_function,
                                                             M=M,
                                                             N=N,
                                                             simple=simple,
                                                             riskless_instruments=riskless_instruments,
                                                             risky_instruments=risky_instruments,
                                                             a_m=a_m,
                                                             b_m=b_m,
                                                             c_i=c_i,
                                                             d_i=d_i,
                                                             LGD=LGD,
                                                             fixed_param=fixed_param,
                                                             pricer=pricer,
                                                             scale=scale)

    if bounds is None:
        bounds = dict(shift=(-0.1, 0.1),
                      alpha=(0.001, 3),
                      theta=(0, 0.1),
                      sigma=(0.001, 0.25),
                      x0=(0.001, 0.1),
                      k=(0.001, 3),
                      nu=(0.001, 0.25),
                      rho=(-0.99, 0.99),
                      y0=(-0.1, 0.1),
                      LGD=(0, 1),
                      spread=(-0.1, 0.1))

    bounds = handler.generate_bounds(**bounds)

    calibration_object = DifferentialEvolution()
    calibration_object.set_bounds(bounds)
    calibration_object.generate_initial_population(size=size, seed=gen_seed)

    start = time.time()
    res_evo = calibration_object.optimize(func=optimization_fun,
                                          generations=generations,
                                          tol=tol,
                                          min_population=min_population,
                                          culling=culling,
                                          generational_cycle=generational_cycle,
                                          seed=opt_seed
                                          )

    # Final optimization
    optimization_fun, handler = define_optimization_function(difference_function,
                                                             M=M,
                                                             N=N,
                                                             simple=simple,
                                                             riskless_instruments=riskless_instruments,
                                                             risky_instruments=risky_instruments,
                                                             a_m=a_m,
                                                             b_m=b_m,
                                                             c_i=c_i,
                                                             d_i=d_i,
                                                             LGD=final_LGD,
                                                             fixed_param=fixed_param,
                                                             pricer=pricer,
                                                             scale=scale)

    bounds = handler.generate_bounds()
    x0 = np.append(res_evo.x[0], 0.4)

    res = minimize(optimization_fun,
                   x0=res_evo.x[0],
                   bounds=bounds,
                   method='L-BFGS-B')

    end = time.time()
    duration = np.round(end - start, 2)
    print('Calibration took {} seconds.'.format(duration))
    print(res)
    print(handler.generate_param_dicts(res.x))

    if return_handler:
        return res, res_evo, calibration_object, duration, handler

    else:
        return res, res_evo, calibration_object, duration

"""
M = 0
N = 2

a = ParamHandler(M=M,
                 N=N,
                 a_m=None,
                 b_m=None,
                 c_i=[1, 1],
                 d_i=[1, 0],
                 LGD=0.4)

b = np.array([0.02, 0.01, 2, 1, 0.04, 0.05, -0.9, 0.06, 0.07])

c, d = a.generate_param_dicts(b)
"""



