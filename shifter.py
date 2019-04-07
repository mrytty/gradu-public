#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from dcf import dcf


def ExtendingPricer(today,
                    BondPriceFunction=None,
                    ZBCPricerFunction=None,
                    InDefaultFunction=None,
                    shift=None,
                    curve=None,
                    default=None,
                    bond_dcf='Act360',
                    ZBC_dcf='Act360'):
    """
    Returns shifted pricing functions
    """

    if shift:

        def bond_func(maturity, **params):

            maturity_in_yrs = dcf(today, maturity, method=bond_dcf)

            # Here shift is purposely different than delta, which should be zero, as we prefer to do shift by
            # using the extension with his shift parameter.
            return BondPriceFunction(0, maturity_in_yrs, **params) * np.exp(-params['shift'] * maturity_in_yrs)

    elif curve is not None:

        zero_curve = curve.interpolate()['Zeros']
        bond_dcf = curve.dcf

        def bond_func(maturity, **params):

            return zero_curve.loc[maturity]

    elif default is not None:

        def bond_func(maturity, **params):

            maturity_in_yrs = dcf(today, maturity, method=bond_dcf)

            # Here we do not use shift method but we use the intrisitic bond value function.
            return BondPriceFunction(0, maturity_in_yrs, **params)

    else:

        raise ValueError('Check inputs.')

    def ZBC_func(strike_date, bond_maturity, K, **params):

        market_to_strike = bond_func(strike_date, **params)
        market_to_bond_maturity = bond_func(bond_maturity, **params)

        S_bond = dcf(today, strike_date, method=bond_dcf)
        T_bond = dcf(today, bond_maturity, method=bond_dcf)

        S_opt = dcf(today, strike_date, method=ZBC_dcf)
        T_opt = dcf(today, bond_maturity, method=ZBC_dcf)

        ref_model_strike = BondPriceFunction(0, S_bond, **params)
        ref_model_bond_maturity = BondPriceFunction(0, T_bond, **params)

        modifier = market_to_bond_maturity / ref_model_bond_maturity

        modified_strike = K * (market_to_strike / ref_model_strike) / modifier

        return ZBCPricerFunction(0, S_opt, T_opt, modified_strike, **params) * modifier

    if InDefaultFunction is None:

        return bond_func, ZBC_func, None

    else:

        def in_default_func(maturity, **params):

            maturity_in_yrs = dcf(today, maturity, method=bond_dcf)

            # Here we do not use shift method but we use the intrisitic bond value function.
            return InDefaultFunction(0, maturity_in_yrs, **params)

        return bond_func, ZBC_func, in_default_func
