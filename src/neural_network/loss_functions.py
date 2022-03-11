#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:42:17 2021

@author: philippbst
"""

import torch


'''

contains general loss function definitions

'''

#%% General loss function implementations

# define the loss function for purely supervised training
def MSEloss(u, u_hat):
    numOutputs = u.shape[0] * u.shape[1]
    MSE = torch.sum((u-u_hat)**2)/numOutputs
    return MSE


# define the loss function for purely supervised training
def MAEloss(u, u_hat):
    numOutputs = u.shape[0] * u.shape[1]
    MSE = torch.sum(torch.abs((u-u_hat)))/numOutputs
    return MSE
