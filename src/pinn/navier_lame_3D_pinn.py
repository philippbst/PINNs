#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:43:58 2021

@author: philippbst
"""

from pinn.general_pinn import GeneralPINN
import torch
from torch.autograd import grad
from pinn.utils_pinn import check_gradientTracking


class NavierLame3DPINN(GeneralPINN):

    def evaluateResiduals(self, X):
        u_pred_dict, u_deriv_dict = self._forwardWithGrads(X)
        Rx, Ry, Rz = self._calculateResiduals(u_pred_dict, u_deriv_dict)
        return Rx, Ry, Rz

    def _calculateResiduals(self, u_pred_dict, u_deriv_dict):
        # unpack PDE parameters and partial derivatives
        lambd = self.pdeParam.lambd
        my = self.pdeParam.my
        omega = self.pdeParam.omega
        rho = self.pdeParam.rho

        u_pred = u_pred_dict.get('u_pred')
        v_pred = u_pred_dict.get('v_pred')
        w_pred = u_pred_dict.get('w_pred')

        u_xx = u_deriv_dict.get('u_xx')
        u_yy = u_deriv_dict.get('u_yy')
        u_zz = u_deriv_dict.get('u_zz')
        u_xy = u_deriv_dict.get('u_xy')
        u_xz = u_deriv_dict.get('u_xz')

        v_xx = u_deriv_dict.get('v_xx')
        v_yy = u_deriv_dict.get('v_yy')
        v_zz = u_deriv_dict.get('v_zz')
        v_yx = u_deriv_dict.get('v_yx')
        v_yz = u_deriv_dict.get('v_yz')

        w_xx = u_deriv_dict.get('w_xx')
        w_yy = u_deriv_dict.get('w_yy')
        w_zz = u_deriv_dict.get('w_zz')
        w_zx = u_deriv_dict.get('w_zx')
        w_zy = u_deriv_dict.get('w_zy')

        # Residual in x direction
        Rx = (rho*(omega**2)*u_pred + my * (u_xx + u_yy + u_zz) + (lambd + my) * (u_xx + v_yx + w_zx))

        # Residual in y direction
        Ry = (rho*(omega**2)*v_pred + my * (v_xx + v_yy + v_zz) + (lambd + my) * (u_xy + v_yy + w_zy))

        # Residuyl in z direction
        Rz = (rho*(omega**2)*w_pred + my * (w_xx + w_yy + w_zz) + (lambd + my) * (u_xz + v_yz + w_zz))
        return Rx, Ry, Rz

    @check_gradientTracking
    def _forwardWithGrads(self, X):
        #X.requires_grad = True

        # making a prediction for u for the given points X
        U_pred = self.model(X)
        u_pred = U_pred[:, 0:1]
        v_pred = U_pred[:, 1:2]
        w_pred = U_pred[:, 2:3]

        # Compute gradients of u,v,w w.r.t inputs x,y,z
        g_u = grad(u_pred.sum(), X, create_graph=True)[0]
        g_v = grad(v_pred.sum(), X, create_graph=True)[0]
        g_w = grad(w_pred.sum(), X, create_graph=True)[0]

        # compute sevond order derivatives of output w.r.t inputs x,y,z
        shape = g_u[:, 0:1].shape
        gg_u_x = grad(g_u[:, 0:1], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_u_y = grad(g_u[:, 1:2], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_u_z = grad(g_u[:, 2:3], X, grad_outputs=torch.ones(shape), create_graph=True)[0]

        gg_v_x = grad(g_v[:, 0:1], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_v_y = grad(g_v[:, 1:2], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_v_z = grad(g_v[:, 2:3], X, grad_outputs=torch.ones(shape), create_graph=True)[0]

        gg_w_x = grad(g_w[:, 0:1], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_w_y = grad(g_w[:, 1:2], X, grad_outputs=torch.ones(shape), create_graph=True)[0]
        gg_w_z = grad(g_w[:, 2:3], X, grad_outputs=torch.ones(shape), create_graph=True)[0]

        # Extract the needed derivatives
        u_xx = gg_u_x[:, 0:1]
        u_xy = gg_u_x[:, 1:2]
        u_xz = gg_u_x[:, 2:3]
        u_yy = gg_u_y[:, 1:2]
        u_zz = gg_u_z[:, 2:3]

        v_xx = gg_v_x[:, 0:1]
        v_yx = gg_v_y[:, 0:1]
        v_yy = gg_v_y[:, 1:2]
        v_yz = gg_v_y[:, 2:3]
        v_zz = gg_v_z[:, 2:3]

        w_xx = gg_w_x[:, 0:1]
        w_yy = gg_w_y[:, 1:2]
        w_zx = gg_w_z[:, 0:1]
        w_zy = gg_w_z[:, 1:2]
        w_zz = gg_w_z[:, 2:3]

        # write model predictions and derivatives in a ditionary to obtain uniform output
        u_pred_collection = [u_pred, v_pred, w_pred]
        u_pred_names = ['u_pred', 'v_pred', 'w_pred']

        u_deriv_collection = [u_xx, u_yy, u_zz, v_xx, v_yy, v_zz,
                              w_xx, w_yy, w_zz, v_yx, w_zx, u_xy, w_zy, u_xz, v_yz]
        u_deriv_names = ['u_xx', 'u_yy', 'u_zz', 'v_xx', 'v_yy', 'v_zz', 'w_xx',
                         'w_yy', 'w_zz', 'v_yx', 'w_zx', 'u_xy', 'w_zy', 'u_xz', 'v_yz']

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
        numEvals = u_pred.shape[0]

        #calculate the residuals
        Rx, Ry, Rz = self._calculateResiduals(u_pred_dict, u_deriv_dict)

        # MSE of Residual in x-direction
        MSE_Rx = torch.sum(Rx**2) / numEvals

        # MSE of Residual in y direction
        MSE_Ry = torch.sum(Ry**2) / numEvals

        # MSE of Residuyl in z direction
        MSE_Rz = torch.sum(Rz**2) / numEvals

        # Entire PDE residual
        MSE_R = MSE_Rx+MSE_Ry+MSE_Rz
        return MSE_R
    

    def _neumannBCLoss(self, X, Y, N = None):
        raise ("Neumann BC not implemented for 3D Navier Lame")

    
    def _robinBCLoss(self, X, Y, N = None):
        raise ("Robin BC not implemented for 3D Navier Lame")









