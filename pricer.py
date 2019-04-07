#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dcf import  dcf as dcf_function
from dcf import implemented_dcf
from affine_model import call as affine_call_price_function
from affine_model import bond_pricer_simple as affine_bond_price_function
from affine_model_with_default import bond_pricer_simple as bond_pricer_default
from affine_model_with_default import defaultable_bond_pricer_simple_with_recovery as default_pricer
from shifter import ExtendingPricer
from derivative import ZeroCouponBond, CallOnZeroCouponBond, PutOnZeroCouponBond, Caplet, Floorlet, Cap, Floor
from G2 import call_on_zero_coupon_bond as G2_call_price_function


class UrPricer:
    
    def __init__(self,
                 today,
                 dcf=None,
                 **kwargs):
        
        self._today = today

        if dcf is None:
            self._dcf = dict(Rate='Act360', Bond='Act360', ZBC='Act360')

        elif dcf in implemented_dcf:
            dcf1 = implemented_dcf[dcf]
            self._dcf = dict(Rate=dcf1, Bond=dcf1, ZBC=dcf1)

        elif isinstance(dcf, dict):
            self._dcf = dcf

        else:
            raise TypeError

    def price(self, instrument, **params):
        """
        This functions takes an instrument, calls on its properties (maturity etc)
        and passes them with parameter dictionary to the pricer function
        """

        if isinstance(instrument, ZeroCouponBond):

            return self._ZeroPricer(instrument.maturity,
                                    **params)

        elif isinstance(instrument, CallOnZeroCouponBond):

            return self._ZBCPricer(instrument.strike_date,
                                   instrument.maturity,
                                   instrument.strike_price,
                                   **params)

        elif isinstance(instrument, PutOnZeroCouponBond):

            call_price = self.price(instrument.call, **params)
            bond_to_strike = self.price(instrument.bond_to_strike, **params)
            underlying_bond = self.price(instrument.underlying_bond, **params)

            return call_price - underlying_bond + bond_to_strike * instrument.strike_price

        elif isinstance(instrument, Caplet) or isinstance(instrument, Floorlet):

            dcf_modifier = dcf_function(instrument.strike_date, instrument.maturity, method=self._dcf['Rate'])
            k_star = 1 + dcf_modifier * instrument.strike_rate

            if isinstance(instrument, Caplet):

                equivalent_option = PutOnZeroCouponBond(market_price=None,
                                                        maturity=instrument.maturity,
                                                        principal=instrument.principal,
                                                        strike_date=instrument.strike_date,
                                                        strike_price=1/k_star)

            else:

                equivalent_option = CallOnZeroCouponBond(market_price=None,
                                                         maturity=instrument.maturity,
                                                         principal=instrument.principal,
                                                         strike_date=instrument.strike_date,
                                                         strike_price=1/k_star)

            return k_star * self.price(equivalent_option, **params)

        elif isinstance(instrument, Cap) or isinstance(instrument, Floor):

            if isinstance(instrument, Cap):
                let = Caplet

            else:
                let = Floorlet

            dates = instrument.coupon_dates + [instrument.maturity]

            lets = [let(market_price=None,
                        maturity=dates[i + 1],
                        principal=instrument.principal,
                        strike_date=dates[i],
                        strike_rate=instrument.strike_rate) for i in range(len(instrument.coupon_dates))]

            return sum([self.price(let, **params) for let in lets])


class G2Pricer(UrPricer):
    """
    Only for G2+++
    Fast, as it does not use analytical formulas
    """

    def __init__(self,
                 today,
                 dcf=None,
                 curve=None,
                 **kwargs):

        super().__init__(today, dcf=dcf, **kwargs)

        self._curve = curve
        self._zeros = self._curve.interpolate()['Zeros']

        def bond_func(maturity, **params):

            return self._zeros[maturity]

        def call_func(strike_date,
                      maturity,
                      strike_price,
                      **params):

            return G2_call_price_function(today=self._today,
                                          strike_date=strike_date,
                                          maturity=maturity,
                                          strike_price=strike_price,
                                          zeros=self._zeros,
                                          dcf_method=self._dcf['Bond'],
                                          **params)

        self._ZeroPricer = bond_func
        self._ZBCPricer = call_func

    @property
    def curve(self):
        return self._curve


class CurvePricer(UrPricer):
    """
    ++ models, needs a curve object
    """
    
    def __init__(self,
                 today,
                 dcf=None,
                 curve=None,
                 bond_price_function=affine_bond_price_function,
                 call_price_function=affine_call_price_function,
                 **kwargs):
        
        super().__init__(today, dcf=dcf, **kwargs)
        
        self._curve = curve

        self._ZeroPricer, self._ZBCPricer, _ = ExtendingPricer(today,
                                                               BondPriceFunction=bond_price_function,
                                                               ZBCPricerFunction=call_price_function,
                                                               curve=self._curve,
                                                               bond_dcf=self._dcf['Bond'],
                                                               ZBC_dcf=self._dcf['ZBC'])

    @property
    def curve(self):
        return self._curve


class SimplePricer(UrPricer):
    """
    + methods
    """
    
    def __init__(self,
                 today,
                 dcf=None,
                 bond_price_function=affine_bond_price_function,
                 call_price_function=affine_call_price_function,
                 **kwargs):

        super().__init__(today, dcf=dcf, **kwargs)

        self._ZeroPricer, self._ZBCPricer, _ = ExtendingPricer(today,
                                                               BondPriceFunction=bond_price_function,
                                                               ZBCPricerFunction=call_price_function,
                                                               shift=True,
                                                               bond_dcf=self._dcf['Bond'],
                                                               ZBC_dcf=self._dcf['ZBC'])


class DefaultPricer(UrPricer):
    """
    Pricer for default models
    """

    def __init__(self,
                 today,
                 dcf=None,
                 bond_price_function=default_pricer,
                 call_price_function=affine_call_price_function,
                 in_default_function=default_pricer,
                 **kwargs):

        super().__init__(today, dcf=dcf, **kwargs)

        self._ZeroPricer, self._ZBCPricer, self._ZeroPricer = ExtendingPricer(today,
                                                                                   BondPriceFunction=bond_price_function,
                                                                                   ZBCPricerFunction=call_price_function,
                                                                                   InDefaultFunction=in_default_function,
                                                                                   default=True,
                                                                                   bond_dcf=self._dcf['Bond'],
                                                                                   ZBC_dcf=self._dcf['ZBC'])

# Testing

"""
yrs = [1,2,3,5,7,10,12,15,20,25,30]

days = [ datetime.date( day=1, month=1, year=2019+i ) for i in yrs ]
swap_par_rate_raw = [4.20, 4.30, 4.70, 5.40, 5.70, 6.00, 6.10, 5.90, 5.60, 5.55, 5.5]
swap_rates = np.array( swap_par_rate_raw ) / 100

data = pd.DataFrame( data={'Rates': swap_rates}, index=days )        
        
today = datetime.date(year=2016,month=1,day=1)

today = datetime.date(day=1, month=1, year=2019)

test_dates = [datetime.date( day=1, month=6, year=2019+i ) for i in yrs[:-1]]

a = LinearCurve(today=today)
a.give_rates(data['Rates'])
a.interpolate()


b = QuadraticCurve(curve=a)
b.interpolate()

d = CubicCurve(curve=a)

c = CubicSplineCurve(curve=a)
c.interpolate()

test_01 = dict(N=1, M=0, shift=0.02, delta=0.00, Y_0=0.03, k=np.array([0.1]), nu=np.array([0.02]))

curver = CurvePricer(today, curve=c)
bonds = [ZeroCouponBond(maturity=date) for date in test_dates]
prices1 = [curver.price(bond, **test_01) for bond in bonds]

simpler = SimplePricer(today, shift=0.02)
prices2 = [simpler.price(bond, **test_01) for bond in bonds]

curver.price(CallOnZeroCouponBond(maturity=test_dates[4],
                                  strike_date=test_dates[3],
                                  strike_price=0.9), **test_01)

simpler.price(CallOnZeroCouponBond(maturity=test_dates[4],
                                   strike_date=test_dates[3],
                                   strike_price=0.9), **test_01)

curver.price(PutOnZeroCouponBond(maturity=test_dates[4],
                                 strike_date=test_dates[3],
                                 strike_price=0.9), **test_01)
"""
