#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:56:10 2021

@author: philippbst
"""

from dataclasses import dataclass

@dataclass(frozen = True)
class NavierLameParams:
    lambd: float 
    my: float
    rho: float
    omega: float = None
    
@dataclass(frozen = True)
class WaveEquationParams:
    c: float 
    omega: float = None

@dataclass(frozen = True)
class NavierLameDampingParams:
    lambd: float 
    my: float
    rho: float
    alpha: float
    beta: float
    omega: float = None
   
