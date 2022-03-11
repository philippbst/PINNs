#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:23:39 2022

@author: philippbst
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from visualization.visualization_general import plotCubicBoundingBox


def loadComsolDisplacements(pathDict,fileName):
    # Read txt file in list
    pathToData = pathDict.get('pathToRawFEMData')
    pathToFile = pathToData+'/'+fileName
    with open(pathToFile,'r',encoding = 'utf-8') as f:
        data = f.readlines()
    
    return data


class ComsolData(object):
    # constructor
    def __init__(self,data):
        self.rawdata = data
        self.displacementDOF = 3
        
        self.dimension = None
        self.nodeNumber = None
        self.numDOF = None
        self.unit = None
        
        self.frequencyRange = None
        self.nodeKoord = None
        self.nodeDisplacements = None        
        
    # methods
    def ExtractInformation(self):
        # Get dimension 
        dimension = self.rawdata[3]
        split = dimension.split(':')
        self.dimension = int(split[1][:-1])
        
        # Get number of nodes
        nodeNumber = self.rawdata[4]
        split = nodeNumber.split(':')
        self.nodeNumber = int(split[1][:-1])
        
        # Get product of frequency numbers and dimension
        expr = self.rawdata[5]
        split = expr.split(':')
        expr = int(split[1][:-1])
        
        # Get number of degrees of freedom and frequencies
        self.numDOF = self.nodeNumber * self.dimension
        numFrequencies = int(expr / self.displacementDOF)
        
        # Get unit
        unit = self.rawdata[7]
        split = unit.split(':')
        unit = split[1][:-1]
        split2 = unit.split(' ')
        self.unit = split2[-1]
                
        # Get start and end frequency
        freqData = self.rawdata[8].split('@')
        freqStart_raw = freqData[1]
        freqEnd_raw = freqData[len(freqData)-self.dimension]
        freqStart = int(freqStart_raw.split('=')[1].split(',')[0])
        freqEnd = int(freqEnd_raw.split('=')[1].split(',')[0])
        self.frequencyRange = np.linspace(freqStart,freqEnd,numFrequencies)
        
        # Transform data in desired format
        nodeKoord = np.zeros([self.nodeNumber,self.dimension])
        nodeDisplacements = {}
        
        self.nodeKoord = nodeKoord
        self.nodeDisplacements = nodeDisplacements
        
        # Itterate through rows = nodes of FEM mesh
        for i in range(self.nodeNumber): 
        
            nodeDataRaw = self.rawdata[9+i]
            nodeData = nodeDataRaw.split(',')
        
            # Create array for node coordinates with shape per line: [X, Y, Z]
            nodeKoord[i,0] = float(nodeData[0])
            nodeKoord[i,1] = float(nodeData[1])
            if self.dimension == 3: # only in three room directions for solids
                nodeKoord[i,2] = float(nodeData[2])
        
            # Itterate through entries per row = Displacement for all frequencies of the node  
            freqIdx = 0
            for j in range(self.dimension,len(nodeData),self.displacementDOF):    
            
                freq = int(self.frequencyRange[freqIdx])
            
                if i == 0:
                    # Displacements always in three room directions
                    # Create dictionary  once that contains for every frequency the displacement field in form: [u_x, u_y, u_z]
                    nodeDisplacements[freq] =  np.zeros([self.nodeNumber,self.displacementDOF])
                
                # Fill displacement array 
                nodeDisplacements[freq][i,0] = float(nodeData[j])
                nodeDisplacements[freq][i,1] = float(nodeData[j+1])
                nodeDisplacements[freq][i,2] = float(nodeData[j+2])
        
                freqIdx = freqIdx+1
        
        self.nodeKoord = nodeKoord
        self.nodeDisplacements = nodeDisplacements
        
    def ShowAttributes(self):
        print(['rawdata','dimension','nodeNumber','numDOF','frequencyRange','nodeKoord','nodeDisplacements'])

    def ShowMethods(self):
        print(['ShowMethods','ShowAttributes','ExtractInformation'])
       
    def FrequencyCheck(self,frequency):
        if not frequency in self.frequencyRange:
            raise ValueError('No results available for the specified frequency')
                
            