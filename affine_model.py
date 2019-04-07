#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
from scipy import integrate


def beta_m(alpha_m, sigma_m):
    """
    Implements Eq. from Nawalka, Beliaeva, Soto (pg. 427)
    """
    return np.sqrt( alpha_m**2 + 2 * sigma_m**2 ) 


def C_i(tau, **params):
    """
    Implements  Eq. 9.36 from Nawalka, Beliaeva, Soto (pg. 427)
    Need params: k
    """
    x = 1 - np.exp( - params['k'] * tau )
    return x / params['k'] 


def B_m(tau, **params):
    """
    Implements  Eq. 9.35 from Nawalka, Beliaeva, Soto (pg. 427)
    Needs params: alpha, sigma
    """
    # Get beta
    beta = beta_m(params['alpha'], params['sigma'])
    
    # Terms
    exp_term = np.exp( beta * tau ) - 1
    denom = ( beta + params['alpha'] ) * exp_term + 2 * beta
    
    return 2 * exp_term / denom 


def first_part_A(tau, **params):
    """
    Implements first part of Eq. 9.34 from Nawalka, Beliaeva, Soto (pg. 427)
    Needs params: alpha, beta, theta, sigma
    """
    #Get beta
    beta = beta_m(params['alpha'], params['sigma'])
    
    # Terms
    log_term_nom = 2 * beta * np.exp( (beta + params['alpha'])*tau/2 )
    log_term_denom = (beta + params['alpha']) * (np.exp( beta * tau) - 1) + 2 * beta
    coef = 2 * params['alpha'] * params['theta'] / params['sigma']**2
    
    return np.sum( coef * np.log( log_term_nom / log_term_denom ) ) 


def last_part_A(tau, **params):
    """
    Implements last part of Eq. 9.34 from Nawalka, Beliaeva, Soto (pg. 427)
    Needs params: k, nu, rho
    """
    
    # Get C
    C = C_i(tau, **params)
    
    # l = len(params['k'])
    l = params['N'] - params['M']

    # Summation
    part_sum = 0
    for i, j in itertools.product(range(l), range(l)):
        if l == 1:
            rho = 1
        else:
            rho = params['rho'][i][j]
            
        coef = params['nu'][i] * params['nu'][j] / (params['k'][i] * params['k'][j]) * rho
        term = ( 1 - np.exp( - (params['k'][i] + params['k'][j]) * tau ) ) / ( params['k'][i] + params['k'][j] )
        part_sum += coef * ( tau - C[i] - C[j] + term )
        
    return part_sum / 2


def A_fun(tau, **params):
    
    gaussian, non_gaussian = 0, 0
    
    if params['M'] > 0:
        non_gaussian = first_part_A(tau, **params)

    if params['N'] - params['M'] > 0:
        gaussian = last_part_A(tau, **params)
        
    return gaussian + non_gaussian


def H_simple(t, T, **params):
    """
    Implements Eq. 9.32 from Nawalka, Beliaeva, Soto (pg. 426)
    """
    
    return params['delta'] * (T-t) 


def bond_pricer_simple(t, T, **params):
    """
    Implements Eq. 9.31 from Nawalka, Beliaeva, Soto (pg. 426)
    """    
    tau = T - t
    A = A_fun(tau, **params)
    B_term, C_term = 0, 0
    
    if params['M'] > 0:
        B_term = np.sum( B_m(tau, **params) * params['X_0'] )
        
    if params['N'] - params['M'] > 0:
        C_term = np.sum( C_i(tau, **params) * params['Y_0'] )
        
    H = H_simple(t, T, **params)
    
    return np.exp( A - B_term - C_term - H )


def Cstar(s, q, **params):
    """
    Implements Eq. 9.66 from Nawalka, Beliaeva, Soto (pg. 434)
    Need params: k
    """    
    nominator = 1 - q * np.exp( - params['k'] * s )
    
    return nominator / params['k']


def q_i(c1, **params):
    """
    Implements Eq. 9.67 from Nawalka, Beliaeva, Soto (pg. 435)
    Need params: k
    """
    return 1 - params['k'] * c1


def beta1(**params):
    """
    Implements Eq. 9.68 from Nawalka, Beliaeva, Soto (pg. 435)
    Need params: alpha, sigma
    """
    return np.sqrt( params['alpha']**2 + 2*params['sigma']**2 )


def beta2(beta1, **params):
    """
    Implements Eq. 9.68 from Nawalka, Beliaeva, Soto (pg. 435)
    Need params: alpha, (sigma)
    """
    return (beta1 - params['alpha']) / 2


def beta3(beta1, **params):
    """
    Implements Eq. 9.68 from Nawalka, Beliaeva, Soto (pg. 435)
    Need params: alpha, (sigma)
    """
    return (-beta1 - params['alpha']) / 2


def beta4(b1, beta1, **params):
    """
    Implements Eq. 9.68 from Nawalka, Beliaeva, Soto (pg. 435)
    Need params: alpha, sigma
    """
    
    first = - (params['alpha'] + b1 * params['sigma']**2)
    
    nominator = first - beta1
    denominator = first + beta1
    
    return nominator / denominator


def Bstar(s, beta1, beta2, beta3, beta4, **params):
    """
    Implements Eq. 9.65 from Nawalka, Beliaeva, Soto (pg. 434)
    Need params: (alpha), sigma
    """
    
    e = beta4 * np.exp( beta1 * s )
    nominator = beta2 * e - beta3
    denominator = e - 1
    
    return 2 * (nominator / denominator) / params['sigma']**2


def g_simple(d):
    """
    d=1 or d=0 (correspend to 2)
    """
    
    def inner_fun(omega, t, S, T, **params):
        """
        Implements Eq 9.61 from Nawalka, Beliaeva, Soto (pg. 434)
        """
        # i
        i = np.array(0+1j)
        
        if d == 1:
            bond_price= bond_pricer_simple(t, T, **params)
            
        elif d == 0:
            bond_price= bond_pricer_simple(t, S, **params)
            
        H1 = H_simple(S, T, **params)
        H2 = H_simple(t, S, **params)
        
        U = T - S
        s = S - t
        
        # Boundary conditions, equantion 9.63
        AU = A_fun(U, **params)
        a1 = AU * (d+i*omega)
        
        # Equation 9.64 (pg. 434)
        A_sum = a1
    
        
        C_solution = 0
        B_solution = 0

        l = params['N'] - params['M']
        
        if l > 0:
            
            # l = len(params['k'])
            
            # Boundary conditions, equantion 9.63
            CU = C_i(U, **params)
            c1 = CU * (d+i*omega)
            
            q1 = q_i(c1, **params)
            
            C_solution = np.sum( Cstar(s, q1, **params) * params['Y_0'] )
            
            C0 = C_i(s, **params)
            
            for i,j in itertools.product(range(l), range(l)):
                
                if l == 1:
                    rho = 1
                else:
                    rho = params['rho'][i][j]
                    
                nominator = params['nu'][i] * params['nu'][j] * rho
                denom = params['k'][i] * params['k'][j]
                
                partial_sum = s - q1[i]*C0[i] - q1[j]*C0[j] + \
                    q1[i]*q1[j] * ( 1 - np.exp( -(params['k'][i] + params['k'][j])*s ) )  / ( params['k'][i] + params['k'][j] )
            
                A_sum += (nominator / denom * partial_sum) / 2
            
        if params['M'] > 0:
            # Boundary conditions, equantion 9.63
            b1 = B_m(U, **params) * (d+i*omega)
            betas1 = beta1(**params)
            betas2 = beta2(betas1, **params)
            betas3 = beta3(betas1, **params)
            betas4 = beta4(b1, betas1, **params)
            B_solution = np.sum( Bstar(s, betas1, betas2, betas3, betas4, **params) * params['X_0'] )
            
            beta_part = betas3 * s + np.log( (1-betas4*np.exp(betas1*s)) / (1-betas4) )
            
            A_sum += - 2 * np.sum( params['alpha'] * params['theta'] / params['sigma']**2 * beta_part )
            
        expo = A_sum - B_solution - C_solution - (H1 * (1 + d*omega)) - H2
                               
        return np.exp(expo) / bond_price
    
    return inner_fun


g1_simple = g_simple(1)
g2_simple = g_simple(0)


def transformation(fun, t, S, T, K, **params):
    
    i = np.array(0+1j)
    
    def inner_fun(x):
        
        f = fun(x, t, S, T, **params)
        ft = np.exp(-i*x*np.log(K)) * f / (i*x)
        
        return np.real(ft)
    
    return inner_fun


def pi1(t, S, T, K, **params):
    fun1 = transformation(g1_simple, t, S, T, K, **params)
    
    I = integrate.quad(fun1, 10**-6, 1000)
    
    return (0.5 + I[0] / np.pi ), I[1]


def pi2(t, S, T, K, **params):
    fun2 = transformation(g2_simple, t, S, T, K, **params)
    
    I = integrate.quad(fun2, 10**-6, 1000)
    
    return (0.5 + I[0] / np.pi), I[1] 


def call(t, S, T, K, **params):

    p1 = pi1(t, S, T, K, **params)[0]
    p2 = pi2(t, S, T, K, **params)[0]

    call_price = bond_pricer_simple(t, T, **params) * p1 - K * bond_pricer_simple(t, S, **params) * p2

    return call_price


def put(t, S, T, K, **params):

    p1 = pi1(t, S, T, K, **params)[0]
    p2 = pi2(t, S, T, K, **params)[0]

    put_price = K * bond_pricer_simple(t, S, **params) * (1 - p2) - bond_pricer_simple(t, T, **params) * (1 - p1)

    return put_price


def call_and_put(t, S, T, K, **params):

    p1 = pi1(t, S, T, K, **params)[0]
    p2 = pi2(t, S, T, K, **params)[0]

    call_price = bond_pricer_simple(t, T, **params) * p1 - K * bond_pricer_simple(t, S, **params) * p2
    put_price = K * bond_pricer_simple(t, S, **params) * (1 - p2) - bond_pricer_simple(t, T, **params) * (1 - p1)

    return p1, p2, call_price, put_price

    








"""
test_01 = dict(N=1, M=0, delta=0.00, Y_0=0.03, k=np.array([0.1]), nu=np.array([0.02]))
test_01a = dict(N=1, M=0, delta=0.03, Y_0=0.03, k=np.array([0.1]), nu=np.array([0.02]))
test_01b = dict(N=1, M=0, delta=0.05, Y_0=0.06, k=np.array([0.1]), nu=np.array([0.05]))
test_02 = dict(N=2, M=0, delta=0.00, Y_0=np.array([0.03,0.02]), k=np.array([0.1,0.15]), nu=np.array([0.02, 0.03]), rho = np.array([[1,0.5],[0.5,1]]))
test_11 = dict(N=1, M=1, delta=0, X_0=0.04, alpha=np.array([0.15]), theta=np.array([0.05]), sigma=np.array([0.05]))
test_13 = dict(N=3, M=1, delta=0, 
               X_0=np.array([0.02]), 
               alpha=np.array([0.2]), 
               theta=np.array([0.04]), 
               sigma=np.array([0.15]),
               Y_0 = np.array([0.02, 0.03]),
               k = np.array([0.1, 0.2]),
               nu = np.array([0.2, 0.25]),
               rho = np.array([[1,0.5],[0.5,1]]))
test_25 = dict(N=5, M=2, delta=0, 
               X_0=np.array([0.02,0.03]), 
               alpha=np.array([0.2,0.1]), 
               theta=np.array([0.04,0.03]), 
               sigma=np.array([0.15,0.2]),
               Y_0 = np.array([0.02, 0.03, 0]),
               k = np.array([0.1, 0.2, 0.3]),
               nu = np.array([0.2, 0.25, 0.1]),
               rho = np.array([[1,0.5,0],[0.5,1,-0.5],[0,-0.5,1]]))

"""

"""
def g2_simple(omega, t, S, T, **params):
    
    #Implements Eq 9.61 from Nawalka, Beliaeva, Soto (pg. 434)
    
    # i
    i = np.array(0+1j)
    
    bond_price2 = bond_pricer_simple(t, S, **params)
    H1 = H_simple(S, T, **params)
    H2 = H_simple(t, S, **params)
    
    U = T - S
    s = S - t
    
    # Boundary conditions, equantion 9.63
    AU = A_fun(U, **params)
    a2 = AU * i * omega
    
    # Equation 9.64 (pg. 434)
    A_sum2 = a2
    
    C_solution2 = 0
    B_solution2 = 0
    
    if params['N'] - params['M'] > 0:
        
        l = len(params['k'])
        
        # Boundary conditions, equantion 9.63
        CU = C_i(U, **params)
        c2 = CU * i * omega
        
        q2 = q_i(c2, **params)
        
        C_solution2 = np.sum( Cstar(s, q2, **params) * params['Y_0'] )
        
        C0 = C_i(s, **params)
        
        for i,j in itertools.product(range(l), range(l)):
            
            if l == 1:
                rho = 1
            else:
                rho = params['rho'][i][j]
                
            nominator = params['nu'][i] * params['nu'][j] * rho
            denom = params['k'][i] * params['k'][j]
            
            partial_sum2 = s - q2[i] * C0[i] - q2[j] * C0[j] + \
            q2[i] * q2[j] * ( 1 - np.exp( -(params['k'][i] + params['k'][j])*s ) ) / ( params['k'][i] + params['k'][j] )
        
            A_sum2 += (nominator / denom * partial_sum2) / 2
        
    if params['M'] > 0:
        # Boundary conditions, equantion 9.63
        b2 = B_m(U, **params) * (i*omega)
        betas1 = beta1(**params)
        betas2 = beta2(betas1, **params)
        betas3 = beta3(betas1, **params)
        betas4 = beta4(b2, betas1, **params)
        B_solution2 = np.sum( Bstar(s, betas1, betas2, betas3, betas4, **params) * params['X_0'] )
        
        beta_part = betas3 * s + np.log( (1-betas4*np.exp(betas1*s)) )/(1-betas4)
        
        A_sum2 -= 2 * np.sum( params['alpha'] * params['theta'] / params['sigma'] * beta_part )
        
    exp2 = A_sum2 - B_solution2 - C_solution2 - (H1 * i*omega) - H2
                           
    return np.exp(exp2) / bond_price2
"""





