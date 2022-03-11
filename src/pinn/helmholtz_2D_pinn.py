#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:36:14 2021

@author: philippbst
"""


from pinn.general_pinn import GeneralPINN
import torch
from torch.autograd import grad
from pinn.utils_pinn import check_gradientTracking
from neural_network.loss_functions import MSEloss


class Helmholtz2DPINN(GeneralPINN):

    def evaluateResiduals(self, X):
        u_pred_dict, u_deriv_dict = self._forwardWithGrads(X)
        R = self._calculateResiduals(u_pred_dict, u_deriv_dict)
        return R

    def robinBC(self, X, Y, N):
        u_pred_dict, u_deriv_dict = self._forwardWithGrads(X)
        
        u_x = u_deriv_dict['u_x'] 
        u_y = u_deriv_dict['u_y'] 
        u = u_pred_dict['u_pred']
        
        # N: 2xn tensor containing surface normal
        
        squared_sum_error_admitt = 0
        for i in range(X.shape[0]):   
            admitt = (u_x[i] * N[i,0] + u_y[i] * N[i,1]) / u[i]
            squared_sum_error_admitt = squared_sum_error_admitt + (admitt - Y[i])**2
        
        MSE_admitt = squared_sum_error_admitt / X.shape[1]
        
        lossBC = MSE_admitt
        
        return lossBC


    def _calculateResiduals(self, u_pred_dict, u_deriv_dict):
        omega = self.pdeParam.omega
        c = self.pdeParam.c

        u_pred = u_pred_dict.get('u_pred')
        u_xx = u_deriv_dict.get('u_xx')
        u_yy = u_deriv_dict.get('u_yy')

        R = (omega**2)*u_pred + (c**2) * (u_xx + u_yy)
        #R = ((omega / c)**2) * u_pred + u_xx + u_yy

        return R

    @check_gradientTracking
    def _forwardWithGrads(self, X):
        # making a prediction for u for the collocation points of the PDE
        U_pred_CP = self.model(X)
        u_pred = U_pred_CP[:, 0:1]

        # Compute gradients of u w.r.t inputs x,y
        g_u = grad(u_pred.sum(), X, create_graph=True)[0]

        # compute sevond order derivatives of output w.r.t inputs x,y
        shape = g_u[:, 0:1].shape

        gg_u_x = grad(g_u[:, 0:1], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_u_y = grad(g_u[:, 1:2], X, grad_outputs=torch.ones(shape), create_graph=True)[0]

        # Extract the needed derivatives
        u_x = g_u[:,0:1]
        u_y = g_u[:,1:2]
        
        u_xx = gg_u_x[:, 0:1]
        u_yy = gg_u_y[:, 1:2]

        # write solutions in dictionary
        u_pred_dict = {}
        u_deriv_dict = {}

        u_pred_dict['u_pred'] = u_pred
        u_deriv_dict['u_x'] = u_x
        u_deriv_dict['u_y'] = u_y
        u_deriv_dict['u_xx'] = u_xx
        u_deriv_dict['u_yy'] = u_yy

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
        self.robinBCLoss(X,Y,N)
        



