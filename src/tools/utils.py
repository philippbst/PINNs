#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:10:06 2021

@author: philippbst
"""

import os
import numpy as np


def getProjectPaths(pathToProject):
    # Set up path structure of the project
    cwd = os.getcwd()
    pathToProject = pathToProject #cwd [:-4] 
    pathNames = ['pathToProject', 'pathToSrc', 'pathToRawFEMData', 'pathToProcessedFEMData'\
                 ,'pathToBMWData','pathToTrainedModels', 'pathToModelRuns', 'pathToModelStudies']
    paths = [pathToProject, pathToProject+'/src', pathToProject+'/data/raw_FEM_data', pathToProject+'/data/processed_FEM_data'\
             ,pathToProject+'/data/BMW_data',pathToProject+'/models/trained_models',pathToProject+'/models/runs',pathToProject+'/models/studies']
    
    pathDict = {}
    for i,name in enumerate(pathNames):
        pathDict[name] = paths[i]

    return pathDict


def scale3dDataPoints(X):
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    x_scaled = (x-min(x)) / (max(x) - min(x))
    y_scaled = (y-min(y)) / (max(y) - min(y)) 
    z_scaled = (z-min(z)) / (max(z) - min(z)) 
    X_scaled = np.array([x_scaled,y_scaled,z_scaled]).T
    
    minmaxDict = {}
    minmaxDict['x_min'] = min(x)
    minmaxDict['y_min'] = min(y)
    minmaxDict['z_min'] = min(z)
    minmaxDict['x_max'] = max(x)
    minmaxDict['y_max'] = max(y)
    minmaxDict['z_max'] = max(z)
    
    return X_scaled,minmaxDict


def rescale3DDataPoints(X,minmaxDict):
    
    x_s = X[:,0]
    y_s = X[:,1]
    z_s = X[:,2]
    
    x_min = minmaxDict['x_min'] 
    y_min = minmaxDict['y_min'] 
    z_min = minmaxDict['z_min'] 
    x_max = minmaxDict['x_max'] 
    y_max = minmaxDict['y_max'] 
    z_max = minmaxDict['z_max'] 

    x_rescaled = x_s * (x_max - x_min) + x_min
    y_rescaled = y_s * (y_max - y_min) + y_min
    z_rescaled = z_s * (z_max - z_min) + z_min
    X_rescaled = np.array([x_rescaled,y_rescaled,z_rescaled]).T
    
    return X_rescaled

def pointsInRange3D(numPoints,x_range,y_range,z_range):
    
    '''
    Takes the range for x,y and z coordinates as input and samples the 
    desired number of points within the domain defined by the coordinate ranges
    '''
    
    r1_norm = np.random.uniform(min(x_range), max(x_range), numPoints)
    r2_norm = np.random.uniform(min(y_range), max(y_range), numPoints)
    r3_norm = np.random.uniform(min(z_range), max(z_range), numPoints)

    pointCloud = np.array([r1_norm,r2_norm,r3_norm]).transpose()

    return pointCloud

def surfPoints(numPoints,range1,range2,fixpoint,fixCoord = 'x'):
    
    '''
    Takes two coordinate ranges as input, a fixed constant and the information 
    which coordinate direction is fixed
    Samples random points in the given ranges with the thired coordinate beeing
    set to the fixed value
    The return array always has the order of:
        r1 = x or y
        r2 = y or z
        depending on the coordinate of the fixed coordinate which can be x,y,z
    
    '''
    
    diff1 = abs(max(range1) - min(range1))
    diff2 = abs(max(range2) - min(range2))
    
    r1 = np.random.rand(numPoints)
    r1_range = ((r1-min(r1)) / (max(r1) - min(r1))) * diff1
    r1_norm = r1_range - abs(min(range1))


    r2 = np.random.rand(numPoints)
    r2_range = ((r2-min(r2)) / (max(r2) - min(r2))) * diff2
    r2_norm = r2_range - abs(min(range2))

    fix = np.ones(numPoints) * fixpoint
    
    if fixCoord == 'x':
        pointCloud = np.array([fix,r1_norm,r2_norm]).transpose()
    elif fixCoord == 'y':
        pointCloud = np.array([r1_norm,fix,r2_norm]).transpose()
    elif fixCoord == 'z':
        pointCloud = np.array([r1_norm,r2_norm,fix]).transpose()
    else:
        raise ValueError
    
    
    return pointCloud

def getSurfPointsOf3DBlock(numSamplesPerSurf,x_range,y_range,z_range):
    '''
    Samples points on all sides of the surface of a rectangular  defined by x,y and z range
    '''
    surf1 = surfPoints(numSamplesPerSurf,x_range,y_range,max(z_range),'z')
    surf2 = surfPoints(numSamplesPerSurf,x_range,y_range,min(z_range),'z')
    surf3 = surfPoints(numSamplesPerSurf,x_range,z_range,max(y_range),'y')
    surf4 = surfPoints(numSamplesPerSurf,x_range,z_range,min(y_range),'y')
    surf5 = surfPoints(numSamplesPerSurf,y_range,z_range,max(x_range),'x')
    surf6 = surfPoints(numSamplesPerSurf,y_range,z_range,min(x_range),'x')
    surfacePoints = np.concatenate((surf1,surf2,surf3,surf4,surf5,surf6),axis = 0) 
    return surfacePoints

