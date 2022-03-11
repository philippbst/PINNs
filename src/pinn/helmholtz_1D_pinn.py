#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:06:02 2021

@author: philippbst
"""

from pinn.general_pinn import GeneralPINN
import torch
from torch.autograd import grad
from pinn.utils_pinn import check_gradientTracking


class Helmholtz1DPINN(GeneralPINN):

    def evaluateResiduals(self, X):
        u_pred_dict, u_deriv_dict = self._forwardWithGrads(X)
        R = self._calculateResiduals(u_pred_dict, u_deriv_dict)
        return R

    def _calculateResiduals(self, u_pred_dict, u_deriv_dict):
        omega = self.pdeParam.omega
        c = self.pdeParam.c

        u_pred = u_pred_dict.get('u_pred')
        u_xx = u_deriv_dict.get('u_xx')

        R = (omega**2)*u_pred + (c**2) * u_xx
        #R = ((omega / c)**2) * u_pred + u_xx
        return R

    @check_gradientTracking
    def _forwardWithGrads(self, X):
        # making a prediction for u for the collocation points of the PDE
        U_pred_CP = self.model(X)
        u_pred = U_pred_CP[:, 0:1]

        # Compute gradients of u w.r.t input x
        g_u = grad(u_pred.sum(), X, create_graph=True)[0]

        # compute sevond order derivatives of output w.r.t input x
        shape = g_u[:, 0:1].shape
        gg_u_x = grad(g_u[:, 0:1], X, grad_outputs=torch.ones(shape), create_graph=True)[0]

        # Extract the needed derivatives
        u_xx = gg_u_x[:, 0:1]

        # write solutions in dictionary
        u_pred_dict = {}
        u_deriv_dict = {}

        u_pred_dict['u_pred'] = u_pred
        u_deriv_dict['u_xx'] = u_xx

        return u_pred_dict, u_deriv_dict

    def _physicsInformedLoss(self, X):
        
        u_pred_dict, u_deriv_dict = self._forwardWithGrads(X)
        
        u_pred = u_pred_dict.get('u_pred')
        numEvals = u_pred.shape[0]

        #calculate the residual
        R = self._calculateResiduals(u_pred_dict, u_deriv_dict)

        # MSE of Residual
        MSE_R = torch.sum(R**2) / numEvals

        return MSE_R

    def _neumannBCLoss(self, X, Y, N = None):
        raise ("Neumann BC not implemented for 1D Helmholtz")
        
    
    def _robinBCLoss(self, X, Y, N = None):
        raise ("Robin BC not implemented for 1D Helmholtz")
