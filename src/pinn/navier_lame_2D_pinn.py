#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:59:00 2021

@author: philippbst
"""


from pinn.general_pinn import GeneralPINN
import torch
from torch.autograd import grad
from pinn.utils_pinn import check_gradientTracking

       
class NavierLame2DPINN(GeneralPINN):
        
    def evaluateResiduals(self, X):
        u_pred_dict, u_deriv_dict = self._forwardWithGrads(X)
        Rx, Ry = self._calculateResiduals(u_pred_dict, u_deriv_dict)
        return Rx, Ry
    
    
    def calculate_displacement_gradient_components(self, X):
        _, u_deriv_dict = self._forwardWithGrads(X)
        return u_deriv_dict.get('u_x'), u_deriv_dict.get('u_y'), u_deriv_dict.get('v_x'), u_deriv_dict.get('v_y')
         
    
    def calculate_linear_strains(self, X):
                
        u_x, u_y, v_x, v_y = self.calculate_displacement_gradient_components(X)

        # Don't creat new tensors and do it in index notation to keep the gradient tracking of autograd
        eps_xx = 0.5 * (u_x + u_x)
        eps_xy = 0.5 * (u_y + v_x)
        eps_yy = 0.5 * (v_y + v_y)
        
        return eps_xx, eps_xy, eps_yy
        
    
    def calculate_cauchy_stresses(self, X):
        
        eps_xx, eps_xy, eps_yy = self.calculate_linear_strains(X)
        
        lambd = self.pdeParam.lambd
        my = self.pdeParam.my
        # Don't creat new tensors and do it in index notation to keep the gradient tracking of autograd
        sigma_xx = lambd * (eps_xx + eps_yy) + 2 * my * eps_xx
        sigma_yy = lambd * (eps_xx + eps_yy) + 2 * my * eps_yy
        sigma_xy = 2 * my * eps_xy
        
        return sigma_xx, sigma_xy, sigma_yy
    
    
    def _neumannBCLoss(self, X, Y, N = None):
        sigma_xx, sigma_xy, sigma_yy = self.calculate_cauchy_stresses(X)

        ''' 
        Highly specific implementation of the Neumann BC loss 
        for the our tackled problem in frequency domain 
        '''
        numEvals = X.shape[0]
        MSE_sig_xy = torch.sum((sigma_xy - Y[:,0])**2) / numEvals
        MSE_sig_yy = torch.sum((sigma_yy - Y[:,1])**2) / numEvals
        loss_BC_Neu = MSE_sig_yy + MSE_sig_xy
        
        ''' 
        TO DO:
        Find a way to generalize implementation and calculate the Neumann BC loss based on Inputs
        Therefore ise sigma_xx, sigma_xy and sigma_yy as input Y to enable calculation of any arbitrary Neumann BC
        '''
        return loss_BC_Neu
    
    
    def _calculateResiduals(self, u_pred_dict, u_deriv_dict):
        # unpack PDE parameters and partial derivatives
        lambd = self.pdeParam.lambd
        my = self.pdeParam.my
        omega = self.pdeParam.omega
        rho = self.pdeParam.rho

        u_pred = u_pred_dict.get('u_pred')
        v_pred = u_pred_dict.get('v_pred')

        u_xx = u_deriv_dict.get('u_xx')
        u_yy = u_deriv_dict.get('u_yy')
        u_xy = u_deriv_dict.get('u_xy')

        v_xx = u_deriv_dict.get('v_xx')
        v_yy = u_deriv_dict.get('v_yy')
        v_yx = u_deriv_dict.get('v_yx')

        ''' Navier Lame in frequency domain'''
        # Residual in x direction
        Rx = (rho*(omega**2)*u_pred + my * (u_xx + u_yy) + (lambd + my) * (u_xx + v_yx))

        # Residual in y direction
        Ry = (rho*(omega**2)*v_pred + my * (v_xx + v_yy) + (lambd + my) * (u_xy + v_yy))
        

        ''' Navier Lame for static case'''
        # # Residual in x direction
        # Rx = my * (u_xx + u_yy) + (lambd + my) * (u_xx + v_yx)

        # # Residual in y direction
        # Ry = my * (v_xx + v_yy) + (lambd + my) * (u_xy + v_yy)

        return Rx, Ry


    @check_gradientTracking
    def _forwardWithGrads(self, X):

        # making a prediction for u for the given points X
        U_pred = self.model(X)
        u_pred = U_pred[:, 0:1]
        v_pred = U_pred[:, 1:2]

        # Compute gradients of u,v,w w.r.t inputs x,y,z
        # g_u = grad(u_pred.sum(), X, create_graph=True)[0]
        # g_v = grad(v_pred.sum(), X, create_graph=True)[0]

        g_u = grad(u_pred, X, grad_outputs=torch.ones(u_pred.shape), create_graph=True)[0]
        g_v = grad(v_pred, X, grad_outputs=torch.ones(u_pred.shape), create_graph=True)[0]
        

        # compute sevond order derivatives of output w.r.t inputs x,y,z
        shape = g_u[:, 0:1].shape
        gg_u_x = grad(g_u[:, 0:1], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_u_y = grad(g_u[:, 1:2], X, grad_outputs=torch.ones(shape), create_graph=True)[0]

        gg_v_x = grad(g_v[:, 0:1], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_v_y = grad(g_v[:, 1:2], X, grad_outputs=torch.ones(shape), create_graph=True)[0]

        # Extract the required derivatives
        u_x = g_u[:,0:1]
        u_y = g_u[:,1:2]
        
        v_x = g_v[:,0:1]
        v_y = g_v[:,1:2]
        
        u_xx = gg_u_x[:, 0:1]
        u_xy = gg_u_x[:, 1:2]
        u_yy = gg_u_y[:, 1:2]

        v_xx = gg_v_x[:, 0:1]
        v_yx = gg_v_y[:, 0:1]
        v_yy = gg_v_y[:, 1:2]

        # write model predictions and derivatives in a ditionary to obtain uniform output
        u_pred_collection = [u_pred, v_pred]
        u_pred_names = ['u_pred', 'v_pred']

        u_deriv_collection = [u_xx, u_yy, v_xx, v_yy, v_yx, u_xy, u_x, u_y, v_x, v_y]
        u_deriv_names = ['u_xx', 'u_yy', 'v_xx', 'v_yy', 'v_yx', 'u_xy','u_x', 'u_y', 'v_x', 'v_y']

        u_pred_dict = {}
        u_deriv_dict = {}

        for i, name in enumerate(u_pred_names):
            u_pred_dict[name] = u_pred_collection[i]

        for i, name in enumerate(u_deriv_names):
            u_deriv_dict[name] = u_deriv_collection[i]
        return u_pred_dict, u_deriv_dict


    def _physicsInformedLoss(self, X):
        
        u_pred_dict, u_deriv_dict = self._forwardWithGrads(X)
        
        u_pred = u_pred_dict.get('u_pred')
        v_pred = u_pred_dict.get('v_pred')
        numEvals = u_pred.shape[0]

        #calculate the residuals
        Rx, Ry = self._calculateResiduals(u_pred_dict, u_deriv_dict)

        # MSE of Residual in x-direction
        MSE_Rx = torch.sum(Rx**2) / numEvals

        # MSE of Residual in y direction
        MSE_Ry = torch.sum(Ry**2) / numEvals
        
        # Entire PDE residual
        loss_CP = MSE_Rx+MSE_Ry
        
        return loss_CP


    def _robinBCLoss(self, X, Y, N = None):
        raise ("Robin BC not implemented for 2D Navier Lame")








