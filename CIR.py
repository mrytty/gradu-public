#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import ncx2

test = dict(alpha=np.array([0.15]), theta=np.array([0.05]), X_0=np.array([0.04]), sigma=np.array([0.1]))


def beta_fun(**params):
    """
    Implements Eq. 6.21 from Nawalka, Beliaeva, Soto (pg. 244)
    """
    return np.sqrt( params['alpha']**2 + 2 * params['sigma']**2 ) 


def B_func(tau, beta, **params):
    """
    Implements Eq. 6.20 from Nawalka, Beliaeva, Soto (pg. 243)
    """
    
    e = np.exp(beta*tau) - 1
    denominator = (beta + params['alpha']) * e + 2 * beta
    
    return 2*e / denominator


def A_func(tau, beta, **params):    
    """
    Implements Eq. 6.22 from Nawalka, Beliaeva, Soto (pg. 243)
    """
    
    factor = 2 * params['alpha'] * params['theta'] / params['sigma']**2
    nominator = 2 * beta * np.exp( (beta + params['alpha']) * tau / 2 )
    denominator = (beta + params['alpha']) * (np.exp(beta*tau) - 1) + 2 * beta
    
    return factor * np.log( nominator / denominator )


def simple_bond_pricer(t, T, **params):
    tau = T - t
    beta0 = beta_fun(**params)
    expo = A_func(tau, beta0, **params) - B_func(tau, beta0, **params) * params['X_0']
    
    return np.exp(expo)


def CIR_call(t, S, T, K, **params):
    
    s = S - t
    U = T - S
    beta1 = beta_fun(**params)
    beta2 = (params['alpha'] + beta1) / params['sigma']**2
    beta3 = 2 * beta1 / (np.exp(beta1*s) - 1) /  params['sigma']**2
    AU = A_func(U, beta1, **params)
    BU = B_func(U, beta1, **params)
    term = (AU - np.log(K)) / BU
    v1 = 2 * (beta2 + beta3 + BU) * term
    v2 = 2 * (beta2 + beta3) * term 
    
    df = 4*params['alpha']*params['theta']/params['sigma']**2
    nc_nominator = 2 * beta3**2 * params['X_0'] * np.exp(beta1*s)
    nc1 =  nc_nominator / (beta2 + beta3 + BU)
    nc2 =  nc_nominator / (beta2 + beta3)
    
    pi1 = ncx2.cdf(v1, df=df, nc=nc1)
    pi2 = ncx2.cdf(v2, df=df, nc=nc2)
    
    bond1 = simple_bond_pricer(t, T, **params)
    bond2 = simple_bond_pricer(t, S, **params)
    
    return pi1, pi2, (bond1 * pi1) - (K * bond2 * pi2)
