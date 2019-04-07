#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import copy
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from dcf import dcf_generator, day_range
from derivative import ZeroCouponBond

### Register datetime converter for a matplotlib plotting method
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class UrCurve:
        
    def __init__(self, today=None, curve=None, dcf='Act360'):
        
        if isinstance(curve, UrCurve):
            self._today = curve._today
            self._dcf = curve._dcf
            self._initialdata = copy.deepcopy(curve._initialdata)
            self._initialdatatype = curve._initialdatatype
            
        else:
            self._today = self.date_handler(today)
            self._dcf = dcf

    @property
    def dcf(self):
        return self._dcf

    @property
    def today(self):
        return self._today
            
    def date_handler(self, date):
        """
        Takes care so that the dates are in datetime.date
        """
        if isinstance(date, np.datetime64):
            return date.astype(datetime.datetime)

        elif isinstance(date, pd.Timestamp):
            return date.date()
        
        else:
            return date
            
    def year_fractions(self, data, method=None):
        """
        Takes given dates from pandas index and calcute yearly fractions.
        So it does not compute the intermediate values.
        Return pd.Series object with yearly fractions (given dcf method)
        """
        
        if method is None:
            method = self._dcf
        
        dates = [self.date_handler(date) for date in data.index]
        fractions = [dcf for dcf in dcf_generator(self._today, dates, method=self._dcf)]
        return pd.Series(fractions, index=dates, name='Years')

    def give_rates(self, rates):
        """
        Takes pd.Series or pd.DataFrame with rates and calculate dcf for dates and makes the initial pd.DataFrame
        """
        self._initialdatatype = 'Rates'
        self._initialdata = pd.DataFrame(rates)
        
    def give_zeros(self, zeros):
        """
        Takes pd.Series or pd.DataFrame with zeros and calculate dcf for dates and makes the initial pd.DataFrame
        """
        self._initialdatatype = 'Zeros'
        self._initialdata = pd.DataFrame(zeros)
        
    def rate_zero_bijection(self, data=None, by=None, method=None):
        """
        Calculates zeros/rates from initial data given dcf method
        Returns a pd.DataFrame object containing, Year fractions, Rates and Zeros
        """
        
        # If nothing is given, then use initial values
        if data is None:
            data = self._initialdata
            
        if by is None:
            by = self._initialdatatype
        
        if method is None:
            method = self._dcf
        
        fractions = self.year_fractions(data, method=method)
        df = pd.concat([fractions, data], axis=1)
        
        if self._initialdatatype == 'Rates':
                zeros = np.exp(- df['Rates'] * df['Years']).dropna()
                df['Zeros'] = zeros
            
        elif self._initialdatatype == 'Zeros':
                rates = - np.log(df['Zeros']) / df['Years'].dropna()
                df['Rates'] = rates
                
        return df
            
    def interpolation(self, data=None, dates=None, method=None, by='Rates'):
        
        if method is None:
            method = self._dcf
            
        if data is None:
            data = self._initialdata

        # set x
        x = self.year_fractions(data, method=method)

        if not isinstance(data, pd.DataFrame):
            raise TypeError

        elif by not in data.columns:
            data = self.rate_zero_bijection(data=data, by=by, method=method)

        # Set y
        if by == 'log-discount':
            y = np.log(data['Zeros'])

        else:
            y = data[by]
            
        # Set interpolator
        interpolator = self._interpolator_function(x, y, **self._interpolator_args)
        
        # Interpolate for these dates
        idx = [date for date in dates]
        xs = [xs for xs in dcf_generator(self._today, idx, method=method)]
        
        # The actual interpolation
        new_values = interpolator(xs)
        
        # Then we calculate Rates and Zeros
        if by == 'Rates':
            rates = new_values
            zeros = np.exp(- rates * xs)
        elif by == 'Zeros':
            zeros = new_values
            rates = - np.log(zeros) / xs
        elif by == 'log-discount':
            zeros = np.exp(new_values)
            rates = - np.log(zeros) / xs
                   
        rates = pd.Series(rates, index=idx)
        zeros = pd.Series(zeros, index=idx)
        years = pd.Series(xs, index=idx)
        
        new_data = pd.DataFrame({'Rates': rates, 'Zeros': zeros, 'Years': years})
        new_data.update(data, overwrite=True)
        
        return new_data
        
    def interpolate(self, method=None, by='Rates'):

        if method is None:
            method = self._dcf

        # Day range generator
        gen = day_range(min(self._initialdata.index), max(self._initialdata.index))

        # Interpolation
        df = self.interpolation(data=self._initialdata, dates=gen, method=method, by=by)

        # Add Instant forward rates
        df = self.add_forward_rates(df)
            
        return df
        
    def add_forward_rates(self, data):

        data['Forward rate'] = self.insta_forward_rate(data)
        return data
            
    def insta_forward_rate(self, data):
        """
        Calculation of instant forward rates.
        """
        zeros = data['Zeros']
        fractions = data['Years']
        derivate = - np.log(zeros).diff(-1) / fractions.diff(-1)
        derivate[-1] = derivate[-2]

        return derivate
        
    def plot(self, dates=None, 
                   method=None, 
                   by='Rates',
                   title=None,
                   save=None):
        """
        """
        
        if dates is None:
            data = self.interpolate(method=method, by=by)
        else:
            data = self.interpolation(self._initialdata, dates, method=method, by=by)
            
        initialdata = self.rate_zero_bijection(method=method)

        plt.subplot(211)

        if title is not None:
            plt.title(title)

        plt.plot(data['Zeros'], label='Interpolated discount curve')
        plt.plot(initialdata['Zeros'], 'o', label='Original inputs')
        plt.legend()
        #plt.show()

        plt.subplot(212)

        plt.plot(data['Rates'], label='Interpolated rate curve')
        plt.plot(data['Forward rate'], label='Instantenous forward rate')
        plt.plot(initialdata['Rates'], 'o', label='Original inputs')
        plt.legend()

        if isinstance(save, str):
            plt.savefig(save)
        plt.show()
            
    def ZeroPricer(self, date=None, dates=None, by='Rates', method=None):
        
        if method is None:
            method = self._dcf
            
        if isinstance(date, datetime.date):
            dates=[date]
            
        df = self.interpolation(data=self._initialdata,
                                dates=dates,
                                by=by,
                                method=method)
        
        return np.array(df['Zeros'])

    def generate_zeros(self, dates=None, by='Rates', method=None):

        zeros = self.ZeroPricer(self, dates=dates, by=by, method=method)
        instruments = [None] * len(dates)

        for i, date in enumerate(dates):
            instruments[i] = ZeroCouponBond(market_price=zeros[i], maturity=date, principal=1)

        return instruments

    def add(self, other, method=None, sub=False):

        if method is None:
            method1, method2 = None, None
        elif isinstance(method, tuple):
            method1, method2, *rest = method
        else:
            raise NotImplementedError

        own_rates = self.interpolate(method=method, by='Rates')['Rates']
        other_rates = self.interpolate(method=method, by='Rates')['Rates']

        if sub:
            return own_rates + other_rates
        else:
            return own_rates - other_rates

    def sub(self, other, method=None, add=False):

        return self.add(other, method=method, sub=not add)


class LinearCurve(UrCurve):
    
    def __init__(self, today=None, curve=None, dcf='Act360'):
        
        super().__init__(today=today, curve=curve, dcf=dcf)
        
        # Set interpolation function to by linear interpolator
        self._interpolator_function = interp1d
        self._interpolator_args = {}

        
class QuadraticCurve(LinearCurve):
    
    def __init__(self, today=None, curve=None, dcf='Act360'):
        
        super().__init__(today=today, curve=curve, dcf=dcf)
      
        # Set interpolation function to use quadratic
        self._interpolator_args = dict(kind='quadratic')

        
class CubicCurve(LinearCurve):
    
    def __init__(self, today=None, curve=None, dcf='Act360'):
        
        super().__init__(today=today, curve=curve, dcf=dcf)
    
        # Set interpolation function to use quadratic
        self._interpolator_args = dict(kind='cubic')

        
class CubicSplineCurve(UrCurve):
    
    def __init__(self, today=None, curve=None, dcf='Act360'):
        
        super().__init__(today=today, curve=curve, dcf=dcf)
        
        # Set interpolation function to use quadratic
        self._interpolator_function = CubicSpline
        self._interpolator_args = {}


"""       
yrs = [1,2,3,5,7,10,12,15,20,25,30]

days = [ datetime.date( day=1, month=1, year=2019+i ) for i in yrs ]
swap_par_rate_raw = [4.20, 4.30, 4.70, 5.40, 5.70, 6.00, 6.10, 5.90, 5.60, 5.55, 5.5]
swap_rates = np.array( swap_par_rate_raw ) / 100

data = pd.DataFrame( data={'Rates': swap_rates}, index=days )        
        
today = datetime.date(year=2016,month=1,day=1)

today=datetime.date(day=1, month=1, year=2019)

test_dates = [ datetime.date( day=1, month=6, year=2019+i ) for i in yrs[:-1] ]


    
a = LinearCurve(today=today)
a.give_rates(data['Rates'])
a.interpolate()


b = QuadraticCurve(curve=a)
b.interpolate()

d = CubicCurve(curve=a)

c = CubicSplineCurve(curve=a)
c.interpolate()
"""
