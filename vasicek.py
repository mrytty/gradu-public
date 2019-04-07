#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def V(t, S, T, **params):
    
    first = 1 - np.exp( -params['k']*(T-S) )
    second = 1 - np.exp( -2*params['k']*(S-t) )
    
    return params['nu']**2 * (first/params['k'])**2 * (second/(2*params['k']))


def alt(t, S, T, K, **params):
    
    V0 = V(t,S,T, **params)
    
    p1 = bond_pricer_simple(t,T,**params)
    p2 = bond_pricer_simple(t,S, **params)
    
    ratio = np.log( p1/(p2*K) )
    
    d1 = (ratio+V0/2)/np.sqrt(V0)
    d2 = (ratio-V0/2)/np.sqrt(V0)
    
    return norm.cdf(d1), norm.cdf(d2), p1 * norm.cdf(d1) - p2 * K * norm.cdf(d2)
