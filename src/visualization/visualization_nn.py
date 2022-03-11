#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:10:02 2021

@author: philippbst
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import visualization.visualization_general as visgen
import numpy as np

COLORMAP_1 = cm.turbo
FONT_STYLE = 'Arial'

def plotMultiLearningCurves(Loss, Names, scaling):
    fig, ax = plt.subplots()
    csfont = {'fontname':FONT_STYLE}
    
    ax.set_title('Loss during training',**csfont)
    ax.set_yscale('log')
    ax.set_xlabel(r'$epochs, i$',**csfont)
    ax.set_ylabel(r'$\mathcal{L}$',**csfont)
    ax.grid()
    
    styles = ['-','-.','--',':','--:','-s', '-*']
    
    for i, name in enumerate(Names):
        if scaling:
            ax.plot(range(0, len(Loss[:, i])), (Loss[:, i]/Loss[0, i]), styles[i])
        else:
            ax.plot(range(0, len(Loss[:, i])), Loss[:, i])

    ax.legend(Names)
    
    return fig, ax


# Used to plot the residuals of a model
def plotResidualDistribution(X, R):

    titleStr = 'Total residuals of the PDE'
    fieldname = r'$R_{total}$'
    
    if X.shape[1] == 1:
        x = X[:, 0]
        y = np.zeros_like(X)
        totalResidual = abs(R[:, 0])
        fig, ax = plt.subplots()
        p = ax.scatter(x, y, c=totalResidual, cmap=COLORMAP_1)
        cbar = fig.colorbar(p, shrink=0.6, aspect=8)
        cbar.set_label(fieldname)
        plt.grid()

    elif X.shape[1] == 2:
        x = X[:, 0]
        y = X[:, 1]
        totalResidual = abs(R[:, 0]) + abs(R[:, 1])
        visgen.scatter_2D_points(X, titleStr, totalResidual, cmap = COLORMAP_1, fieldname = fieldname)
        
    elif X.shape[1] == 3:
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        totalResidual = abs(R[:, 0]) + abs(R[:, 1]) + abs(R[:, 2])
        visgen.scatter_3D_points(X, titleStr, totalResidual, cmap = COLORMAP_1, fieldname = fieldname)


    
#%% Currently not in use and old     
def plotDeviationDistribution(X, U_dev):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    totalDeviation = abs(U_dev[:, 0]) + abs(U_dev[:, 1]) + abs(U_dev[:, 2])

    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    p = ax.scatter3D(x, y, z, c=totalDeviation, cmap=cm.coolwarm)
    visgen.plotCubicBoundingBox(ax, x, y, z)
    titleStr = 'Total deviation of model predictions'
    ax.set_title(titleStr)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(p)
    plt.grid()
    plt.show()

    
class NNPredictions(object):
    def __init__(self, U_predict, inputKoord, frequency, dimension, unit):
        self.prediction = U_predict
        self.inputKoord = inputKoord
        self.dimension = dimension
        self.unit = unit
        self.frequency = frequency

        self.deformedKoord = None
        self.deformationAmplification = None

    def CalulateDeformedKoord(self, amplification):
        self.deformedKoord = self.inputKoord + amplification * self.prediction
        self.deformationAmplification = amplification

    def VisualizePrediction(self, amplification):
        if self.deformedKoord == None:
            self.CalulateDeformedKoord(amplification)
        else:
            pass
        x = self.deformedKoord[:, 0]
        y = self.deformedKoord[:, 1]
        z = self.deformedKoord[:, 2]
        totalDeformation = abs(self.prediction[:, 0]) + abs(self.prediction[:, 1]) + abs(self.prediction[:, 2])

        fig = plt.figure()
        ax = plt.axes(projection='3d', adjustable='box')
        ax.scatter3D(x, y, z, c=totalDeformation, cmap=cm.coolwarm)
        visgen.plotCubicBoundingBox(ax, x, y, z)
        titleStr = f"Plot of deformation predicted by the Neural Network at {self.frequency} Hz"
        ax.set_title(titleStr)
        ax.set_xlabel('X ['+self.unit+']')
        ax.set_ylabel('Y ['+self.unit+']')
        ax.set_zlabel('Z ['+self.unit+']')
        plt.grid()
        plt.show()