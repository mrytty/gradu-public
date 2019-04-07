#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import datetime

implemented_dcf = {}


def add_dcf_method(func, method_name):

    def inner_fun(d0, d1):
        """Calculates day count factors"""
        """Assumes that days are given as datetime objects"""

        if d0 == d1:
            return 0
        elif d0 > d1:
            d0, d1 = d1, d0

        return func(d0, d1)

    implemented_dcf[method_name] = inner_fun

    return inner_fun


def bond30360(d0, d1):

    year = 360 * (d1.year - d0.year)
    month = 30 * (d1.month - d0.month)
    adj_day0 = min(d0.day, 30)

    if d1 == 30:

        adj_day1 = min(d1.day, 30)

    day = adj_day1 - adj_day0

    return np.array((year + month + day) / np.array(360.0))


def us30360(d0, d1):

    year = 360 * (d1.year - d0.year)
    month = 30 * (d1.month - d0.month)
    adj_day0, adj_day1 = min(d0.day, 30), d1.day

    if d0.month == 2 and d0.eom():

        if d1.month == 2 and d1.month():
            adj_day1 = 30
        adj_day0 = 30

    if adj_day1 == 31 and adj_day0 in (30, 31):
        adj_day1 = 30
    day = adj_day1 - adj_day0

    return (year + month + day) / np.array(360.0)


def actactisda(d0, d1):

    if d0.year == d1.year:
        delta = d1 - d0

        return delta.days / (np.array(365.0) + is_leap_year(d0))

    else:

        end_of_year = d0.replace(month=12, day=31)
        start_of_year = d1.replace(month=1, day=1)
        yrs = (d1.year - d0.year) - 1
        delta = (end_of_year - d0)
        delta1 = (d1 - start_of_year)

        return yrs + delta.days / (np.array(365.0) + is_leap_year(d0)) + delta1.days / (365.0 + is_leap_year(d1))


def act360(d0, d1):

    diff = d1 - d0

    return diff.days / np.array(360.0)


add_dcf_method(bond30360, '30/360Bond')
add_dcf_method(us30360, '30/360US')
add_dcf_method(actactisda, 'ActActISDA')
add_dcf_method(act360, 'Act360')


def dcf(d0, d1, method='Act360'):
    """Calculates day count factors"""
    """Assumes that days are given as datetime objects"""
        
    return implemented_dcf[method](d0, d1)


def day_range(d0, d1):
    """
    A generator that returns the dates between d0 and d1 (inclusive)
    """
    
    delta = (d1 - d0).days
    d = 0
    while d <= delta:
        yield d0 + datetime.timedelta(days=d)
        d+=1


def dcf_generator(d0, iterable, method='Act360'):
    """
    generator that returns a dcfs between d0 and dates from iterable (of dates)
    """
    
    for date in iterable:
        yield dcf(d0, date, method=method)


def dcf_range(d0, d1, method='Act360'):
    """
    generator that returns a dcfs between for all dates between d1 and d1 (inclusive)
    """
    
    for date in day_range(d0, d1):
        yield dcf(d0, date, method=method)


def is_leap_year(d0):
    """
    Checks if a year is a leap year (True) or not (False)
    """
    # If we are given a datetime.date object, then pick the year
    if isinstance(d0, datetime.date):
        year = d0.year
    
    # Otherwise we assume it is basically an integer.
    else:
        year = int(d0)
        
    if year % 4:
        # if it is not divisable by 4, then it is a common year
        return False
    
    elif year % 100:
        # otherwise, if it is not centennial year, it is a leap year
        return True
    
    elif year % 400:
        # if it is a centennial year not divisible by 400, then it is a leap year
        return False
    
    else:
        return True
