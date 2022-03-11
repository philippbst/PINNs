#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 09:44:32 2021

@author: philippbst
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

COLORMAP_1 = cm.turbo
TITLE_FONT_STYLE = 'Arial'

# For visulaization of 2D and 3D points 
class Visualizer(object):
    def __init__(self, koord, disp, dim):
        self.koord = koord
        self.displacements = disp
        self.dim = dim
    
    
    def plot_with_deformation(self, amplification, title = ''): 
        x_deformed = self.koord[:,0] + amplification * self.displacements[:,0]
        y_deformed = self.koord[:,1] + amplification * self.displacements[:,1]
        fieldname = r'$ \parallel \mathbf{u} \parallel_{2}$' + r'  $[m]$'
        
        if self.dim == 3:
            z_deformed = self.koord[:,2] + amplification * self.displacements[:,2]
            total_displacement = ((self.displacements[:,0])**2 + (self.displacements[:,1])**2 + (self.displacements[:,2])**2)**0.5
            X_deformed = np.array([x_deformed, y_deformed, z_deformed]).T
            fig, ax = scatter_3D_points(X_deformed, title, total_displacement, fieldname)
            
        else:
            total_displacement = ((self.displacements[:,0])**2 + (self.displacements[:,1])**2)**0.5
            X_deformed = np.array([x_deformed, y_deformed]).T
            fig, ax = scatter_2D_points(X_deformed, title, total_displacement, fieldname)
            
        return fig, ax
    

def scatter_3D_points(X, title = '', color = 'r', fieldname = '', alpha = 1, cmap = COLORMAP_1, colorbar = True):
    fig = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    p = ax.scatter3D(X[:,0], X[:,1], X[:,2],c=color, alpha = alpha, cmap=cmap)
    plotCubicBoundingBox(ax,X[:,0],X[:,1],X[:,2]) # to have equal axis
    ax.set_xlabel(r'$x [m]$')
    ax.set_ylabel(r'$y [m]$')
    ax.set_zlabel(r'$z [m]$')
    if colorbar:
        cbar = fig.colorbar(p, shrink=0.6, aspect=8)
        #cbar.ax.set_title(fieldname)
        cbar.set_label(fieldname)
    csfont = {'fontname':TITLE_FONT_STYLE}
    ax.set_title(title,**csfont)
    plt.grid()
    return fig, ax

    
def scatter_2D_points(X, title = '', color = 'r', fieldname = '', alpha = 1, cmap = COLORMAP_1, colorbar = True):
    fig, ax = plt.subplots()
    p = plt.scatter(X[:,0], X[:,1], c=color, alpha = alpha, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel(r'$x [m]$')
    ax.set_ylabel(r'$y [m]$')
    if colorbar:
        cbar = fig.colorbar(p, shrink=0.6, aspect=8)
        #cbar.ax.set_title(fieldname)
        cbar.set_label(fieldname)
    csfont = {'fontname':TITLE_FONT_STYLE}
    ax.set_title(title,**csfont)
    plt.grid()     
    return fig, ax



def plot_interpolated_squared_field(X, U, direction = 'sum', title = 'Interpolated displacementfield', cmap = COLORMAP_1,\
                                    levels = 100, field_name = "", norm_cbar = False, X_norm = None, U_norm = None):
    
    # data coordinates and values
    x = X[:,0]
    y = X[:,1]
    
    # target grid to interpolate to
    xi = np.arange(min(x),max(x),0.001)
    yi = np.arange(min(y),max(y),0.001)
    xi,yi = np.meshgrid(xi,yi)

    # interpolate
    zi = griddata((x,y),U,(xi,yi),method='cubic')

    # plot
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    surf = plt.contourf(xi,yi,zi, cmap = cmap, levels = levels)

    ax.grid('minor')
    csfont = {'fontname':TITLE_FONT_STYLE}
    ax.set_title(title,**csfont)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    
    
    if norm_cbar:
        plt.clim(min(U_norm), max(U_norm)) 
        plot_identical_colorbar(fig, ax, X_norm, U_norm, cmap = cmap, levels = levels, field_name = field_name)
    else: 
        cbar = fig.colorbar(surf, shrink=0.55, aspect=12) # for 2D Navier Lame
        #cbar = fig.colorbar(surf, shrink=0.65, aspect=12) # for 3D Navier Lame QS1
        #cbar = fig.colorbar(surf, shrink=0.48, aspect=12) # for 3D Navier Lame QS2
        
        cbar.set_label(field_name)
        #cbar = fig.colorbar(surf, shrink=0.6, aspect=8, format=ticker.FuncFormatter(fmt))
        #cbar.ax.set_title(field_name)
        #cbar.ax.locator_params(nbins=5)
    

    return fig, ax



# Create a plot for creation of an identical colorbar
def plot_identical_colorbar(fig, ax, X, U, cmap = COLORMAP_1, levels = 100, field_name = ""):
    # data coordinates and values
    x = X[:,0]
    y = X[:,1]
    
    # target grid to interpolate to
    xi = np.arange(min(x),max(x),0.001)
    yi = np.arange(min(y),max(y),0.001)
    xi,yi = np.meshgrid(xi,yi)

    # interpolate
    zi = griddata((x,y),U,(xi,yi),method='cubic')

    # plot
    fig_imag, ax_imag = plt.subplots()
    surf = plt.contourf(xi,yi,zi, cmap = cmap, levels = levels)
    #divider = make_axes_locatable(ax_imag)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cbar = fig.colorbar(surf, shrink=0.55, aspect=12) # for 2D Navier Lame
    #cbar = fig.colorbar(surf, shrink=0.65, aspect=12) # for 3D Navier Lame QS1
    #cbar = fig.colorbar(surf, shrink=0.48, aspect=12) # for 3D Navier Lame QS2
    
    cbar.set_label(field_name)
    
    #cbar.ax.set_title(field_name)
    plt.close(fig_imag)
   
    
# can be used for scientific notation of the colorbar
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)



def plotCubicBoundingBox(ax,X,Y,Z):
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], c = 'b')
           
           
          
           
           
           