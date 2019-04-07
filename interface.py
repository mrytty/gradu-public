import os
import numpy as np
import pandas as pd
import time
import pickle
import copy

from affine_model import bond_pricer_simple as affine_bond_price_function

from affine_model import call as affine_call_price_function
from affine_model_with_default import bond_pricer_simple as bond_pricer_default
from affine_model_with_default import defaultable_bond_pricer_simple_with_recovery as default_pricer

from curve import UrCurve, LinearCurve, QuadraticCurve, CubicCurve, CubicSplineCurve
from pricer import CurvePricer, SimplePricer, DefaultPricer
from calibration import squared, relative_squared, own_calibration, array_to_param
from calibration_with_default import own_calibration as own_calibration_default
from calibration_with_default import ParamHandler, ParamHandlerWithFixed

from data import date_handler


class Calibrator:
    """
    Class for handling manual tasks of calibration
    """

    def __init__(self,
                 today=None,
                 M=0,
                 N=1,
                 riskless_curve=None,
                 pricer='simple',
                 pricer_dcf=None,
                 *args,
                 **kwargs):

        self._riskless_curve = None
        self._parameters = None
        self._res = None
        self._res_evo = None

        if today is not None:
            self._today = date_handler(today)

        elif isinstance(riskless_curve, UrCurve):
            self._today = riskless_curve.today

        else:
            raise ValueError

        if pricer == 'simple':
            self._pricer = SimplePricer(self._today,
                                        dcf=pricer_dcf,
                                        bond_price_function=affine_bond_price_function,
                                        call_price_function=affine_call_price_function)

            simple = True
            curve = None

        elif pricer == 'curve':
            if isinstance(self._risklesscurve, UrCurve):
                self._pricer = CurvePricer(self._today,
                                           dcf=pricer_dcf,
                                           curve=self._risklesscurve,
                                           bond_price_function=affine_bond_price_function,
                                           call_price_function=affine_call_price_function)

                simple = False
                curve = self._riskless_curve

            else:
                print('Give also curve data!')
                raise ValueError

        elif pricer == 'default':
            self._pricer = DefaultPricer(self._today,
                                         dcf=pricer_dcf,
                                         bond_price_function=bond_pricer_default,
                                         call_price_function=affine_call_price_function,
                                         in_default_function=default_pricer)

            simple = True
            curve = None

        else:
            raise ValueError

        self.set_parameters(M=M,
                            N=N,
                            simple=simple,
                            curve=curve)

        self._optimization_args = {}
        self.set_optimization_args()

    @property
    def optimizations_args(self):

        return self._optimization_args

    @optimizations_args.setter
    def optimizations_args(self, x):

        if isinstance(x, dict):
            self._optimization_args.update(x)
            self.print_args()

    def print_args(self):

        for key in self._optimization_args:
            print('{}: {}\n'.format(key, self._optimization_args[key]))

    @property
    def riskless_curve(self):

        return self._riskless_curve

    @property
    def parameters(self):

        return self._parameters

    def set_parameters(self, M=0, N=1, simple=True, curve=None):

        parameters = dict(M=M, N=N, simple=simple, curve=curve)

        if self._parameters is None:
            self._parameters = parameters

        return parameters

    def set_optimization_args(self,
                              diff_fun=squared,
                              simple=True,
                              scale=1000000,
                              generations=1000,
                              tol=0.001,
                              initial_population=2048,
                              min_population=32,
                              generational_cycle=10,
                              size=1024,
                              culling=(0.1, 0.3)):

        self._optimization_args = dict(diff_fun=diff_fun,
                                       simple=simple,
                                       scale=scale,
                                       generations=generations,
                                       tol=tol,
                                       initial_population=initial_population,
                                       min_population=min_population,
                                       generational_cycle=generational_cycle,
                                       size=size,
                                       culling=culling)

    def _optimize(self, riskless_instruments=None, risky_instruments=None, gen_seed=None, opt_seed=None):

        M, N = self.parameters['M'], self.parameters['N']
        simple = self.parameters['simple']

        # This is the actual calibration call
        res, res_evo, calibration_object, duration = own_calibration(M=M,
                                                                     N=N,
                                                                     gen_seed=gen_seed,
                                                                     opt_seed=opt_seed,
                                                                     simple=simple,
                                                                     difference_function=self.optimizations_args['diff_fun'],
                                                                     instruments=riskless_instruments,
                                                                     pricer=self._pricer,
                                                                     tol=self.optimizations_args['tol'],
                                                                     generations=self.optimizations_args['generations'],
                                                                     size=self.optimizations_args['size'],
                                                                     scale=self.optimizations_args['scale'],
                                                                     min_population=self.optimizations_args['min_population'],
                                                                     culling=self.optimizations_args['culling'],
                                                                     generational_cycle=self.optimizations_args['generational_cycle'])

        param = array_to_param(res.x, M=M, N=N, simple=simple)

        return param, res, res_evo, calibration_object, duration

    def differences(self, instruments, param):
        # Makes a df that has information about the fit
        info = [[deriv.maturity,
                 deriv.calibrate,
                 self._pricer.price(deriv, **param)] for deriv in instruments]
        info = np.array(info)
        df = pd.DataFrame(info[:, 1:], index=info[:, 0], columns=['Market price', 'Calibrated price'])
        df['Difference'] = df['Market price'] - df['Calibrated price']
        df['Abs difference'] = np.abs(df['Difference'])
        df['Pct difference'] = df['Difference'] / df['Market price'] * 100
        df['Pct abs difference'] = np.abs(df['Pct difference'])

        return df

    def optimize(self, riskless_instruments=None, risky_instruments=None, gen_seed=None, opt_seed=None):

        param, res, res_evo, calibration_object, duration = self._optimize(riskless_instruments=riskless_instruments,
                                                                           risky_instruments=risky_instruments,
                                                                           gen_seed=gen_seed,
                                                                           opt_seed=opt_seed)

        self._res = res
        self._res_evo = res_evo

        df_riskfree = None
        df_risky = None

        param['risk_free'] = True

        param_risky = copy.copy(param)
        param_risky['risk_free'] = False

        # Differences
        if riskless_instruments is not None:
            df_riskfree = self.differences(riskless_instruments, param)

        if risky_instruments is not None:
            df_risky = self.differences(risky_instruments, param_risky)

        d = dict(param=param,
                 riskfree=df_riskfree,
                 risky=df_risky,
                 res=res,
                 res_evo=res_evo,
                 duration=duration)

        return d


class DefaultCalibrator(Calibrator):
    """
    The same but with defaults
    """

    def __init__(self,
                 today=None,
                 Ms=(0, 0, 1),
                 ns=(1, 1, 0),
                 pricer='default',
                 pricer_dcf=None,
                 LGD=None,
                 fixed_param=None,
                 *args,
                 **kwargs):

        # Ms and Ns are tuples (x, y, z) where
            # x is the number of common factors (gaussian/square-root)
            # y is the number of factors unique to risk-free rate (gaussian/square-root)
            # z is the number od factors unique to spread process (gaussian/square-root)
        # Thus the model has x + y + z factors altogether
        # Risk-free process has x + y factors
        # Spread process has y + z factors

        mx, my, mz, *rest = Ms
        nx, ny, nz, *rest = ns

        M = sum([mx, my, mz])
        N = M + sum([nx, ny, nz])

        self._fixed_param = fixed_param

        super().__init__(today=today, M=M, N=N, pricer=pricer, pricer_dcf=pricer_dcf, *args, **kwargs)

        self._parameters['LGD'] = LGD

        self._parameters['a_m'], self._parameters['b_m'] = self.indicator_vector(mx, my, mz)
        self._parameters['c_i'], self._parameters['d_i'] = self.indicator_vector(nx, ny, nz)

    def indicator_vector(self, common, riskfree_unique, spread_unique):

        common_part = [1] * common
        riskfree = common_part + [1] * riskfree_unique + [0] * spread_unique
        spread = common_part + [0] * riskfree_unique + [1] * spread_unique

        return riskfree, spread

    def _optimize(self, riskless_instruments=None, risky_instruments=None, gen_seed=None, opt_seed=None):

        M, N = self.parameters['M'], self.parameters['N']
        simple = self.parameters['simple']
        a_m, b_m = self.parameters['a_m'], self.parameters['b_m']
        c_i, d_i = self.parameters['c_i'], self.parameters['d_i']
        final_LGD = self.parameters['LGD']

        res, res_evo, calibration_object, duration = own_calibration_default(M=M,
                                                                             N=N,
                                                                             gen_seed=gen_seed,
                                                                             opt_seed=opt_seed,
                                                                             simple=simple,
                                                                             LGD=1,
                                                                             final_LGD=final_LGD,
                                                                             difference_function=self.optimizations_args['diff_fun'],
                                                                             riskless_instruments=riskless_instruments,
                                                                             risky_instruments=risky_instruments,
                                                                             pricer=self._pricer,
                                                                             tol=self.optimizations_args['tol'],
                                                                             generations=self.optimizations_args['generations'],
                                                                             size=self.optimizations_args['size'],
                                                                             scale=self.optimizations_args['scale'],
                                                                             min_population=self.optimizations_args['min_population'],
                                                                             culling=self.optimizations_args['culling'],
                                                                             generational_cycle=self.optimizations_args['generational_cycle'],
                                                                             fixed_param=self._fixed_param,
                                                                             a_m=a_m,
                                                                             b_m=b_m,
                                                                             c_i=c_i,
                                                                             d_i=d_i)

        if self._fixed_param is None:
            handler = ParamHandler(M=M,
                                   N=N,
                                   a_m=a_m,
                                   b_m=b_m,
                                   c_i=c_i,
                                   d_i=d_i,
                                   LGD=final_LGD,
                                   simple=simple)

        else:
            handler = ParamHandlerWithFixed(M=M,
                                            N=N,
                                            a_m=a_m,
                                            b_m=b_m,
                                            c_i=c_i,
                                            d_i=d_i,
                                            LGD=final_LGD,
                                            fixed_param=self._fixed_param,
                                            simple=simple)


        param_riskfree, param_defaultable = handler.generate_param_dicts(res.x)

        return param_defaultable, res, res_evo, calibration_object, duration


def mass_calibration(key, data, diff_fun=squared, gen_seed=None, opt_seed=None, results=None, models=None):

    start = time.time()

    if results is None:
        results = {}

    dates = list(data.data.keys())

    results[key] = {}

    for model in models:

        M, N = model
        print('Model: {}'.format(model))

        calibrator = Calibrator(riskless_curve=data.curve(), M=M, N=N)
        calibrator.set_optimization_args(diff_fun=diff_fun)
        results[key][model] = calibrator.optimize(riskless_instruments=data.zeros(dates),
                                                  gen_seed=gen_seed,
                                                  opt_seed=opt_seed)

    end = time.time()
    duration = np.round(end - start, 2)

    print('Whole calibration took {} seconds.'.format(duration))

    return results


def mass_calibration_with_default(key,
                                  riskfree_data,
                                  risky_data,
                                  LGD=1,
                                  diff_fun=squared,
                                  gen_seed=None,
                                  opt_seed=None,
                                  results=None,
                                  models=None,
                                  fixed_param=None):

    start = time.time()

    if results is None:
        results = {}

    riskfree_dates = list(riskfree_data.data.keys())
    risky_dates = list(risky_data.data.keys())

    results[key] = {}

    for model in models:

        Ms, ns = model
        print('Model: {}'.format(model))

        calibrator = DefaultCalibrator(riskless_curve=riskfree_data.curve(),
                                       Ms=Ms,
                                       ns=ns,
                                       LGD=LGD,
                                       fixed_param=fixed_param)

        calibrator.set_optimization_args(diff_fun=diff_fun,
                                         simple=True,
                                         scale=1000000,
                                         generations=500,
                                         tol=0.001,
                                         initial_population=512,
                                         min_population=32,
                                         generational_cycle=5,
                                         size=512,
                                         culling=(0.4, 0.6))

        results[key][model] = calibrator.optimize(riskless_instruments=riskfree_data.zeros(riskfree_dates),
                                                  risky_instruments=risky_data.zeros(risky_dates),
                                                  gen_seed=gen_seed,
                                                  opt_seed=opt_seed)

    end = time.time()
    duration = np.round(end - start, 2)

    print('Whole calibration took {} seconds.'.format(duration))

    return results


def analyze_models_for_item(item, results):
    item_data = results[item]

    relative = {}
    absolute = {}
    param = {}
    durations = {}

    for model in item_data:

        data = item_data[model]

        validity = data['res']['success']

        if validity:
            pass

            df = data['riskfree']
            relative[model] = df['Pct difference']
            absolute[model] = df['Pct abs difference']
            param[model] = data['param']
            durations[model] = data['duration']

    relative = pd.DataFrame(relative)
    absolute = pd.DataFrame(absolute)

    print('Running these calibrations took {} minutes.'.format(round(sum(durations.values()) / 60, 2)))

    relative.plot()
    absolute.plot()

    print('Absolute relative errors by models:')
    print(absolute.mean().sort_values())

    return relative, absolute, param, durations

location_of_pickled_data = 'C:/Users/Miikka/PycharmProjects/gradu/'
data_pickle = 'data.pickle'
def_data_pickle = 'def_data_rel.pickle'

if os.path.isfile(location_of_pickled_data + def_data_pickle):
    with open(location_of_pickled_data + def_data_pickle, 'rb') as f:
        def_results = pickle.load(f)


def analyze_default_models_for_item(item, results):
    item_data = results[item]

    relative_rf = {}
    absolute_rf = {}
    relative_risky = {}
    absolute_risky = {}
    absolute = {}
    param = {}
    durations = {}

    for model in item_data:

        data = item_data[model]

        validity = data['res']['success']

        if validity:

            df_rf = data['riskfree']
            relative_rf[model] = df_rf['Pct difference']
            absolute_rf[model] = df_rf['Pct abs difference']

            df_risky = data['risky']
            relative_risky[model] = df_risky['Pct difference']
            absolute_risky[model] = df_risky['Pct abs difference']

            absolute[model] = (df_rf['Market price'] * df_rf['Pct abs difference'] + df_risky['Market price'] * df_risky['Pct abs difference']) / (df_rf['Market price'] + df_risky['Market price'])

            param[model] = data['param']
            durations[model] = data['duration']

    relative_rf = pd.DataFrame(relative_rf)
    absolute_rf = pd.DataFrame(absolute_rf)

    relative_risky = pd.DataFrame(relative_risky)
    absolute_risky = pd.DataFrame(absolute_risky)

    absolute = pd.DataFrame(absolute)

    print('Running these calibrations took {} minutes.'.format(round(sum(durations.values()) / 60, 2)))

    relative_rf.plot()
    relative_risky.plot()
    absolute_rf.plot()
    absolute_risky.plot()
    absolute.plot()

    print('Absolute relative errors by models for risk-free:')
    print(relative_rf.mean().sort_values())

    print('Absolute relative errors by models for risky:')
    print(relative_risky.mean().sort_values())

    print('Combined absolute relative errors by models:')
    print(absolute.mean().sort_values())

    return absolute, relative_rf, relative_risky, absolute_rf, absolute_risky, param, durations

"""
a1, a2, a3, a4, a5, a6, a7 = analyze_default_models_for_item('ios, swap', def_results)

pricer = DefaultPricer(date_handler(date_string))

for item in def_results:

    riskfree, risky = item.split(', ')

    if riskfree == 'ios':
        riskfree = ios
    elif riskfree == 'germany':
        riskfree = germany

    rf_dates = list(riskfree.data.keys())
    rf_inst = riskfree.zeros(rf_dates)

    if risky == 'italy':
        risky = italy

    elif risky == 'france':
        risky = france

    elif risky == 'swap':
        risky = swap

    risky_dates = list(risky.data.keys())
    risky_inst = risky.zeros(risky_dates)

    for model in def_results[item]:
        data = def_results[item][model]
        param = data['param']
        pricer.price
"""
