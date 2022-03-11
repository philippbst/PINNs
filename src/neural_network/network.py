#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:42:17 2021

@author: philippbst
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from neural_network.loss_functions import MSEloss


# Neural network creation, training and testing routines

# Defining an "ordinary" neural network (MLP) with variable number of layers
class NeuralNetwork(torch.nn.Module):
    def __init__(self, inputSize, outputSize, numLayers, layerSize):
        super(NeuralNetwork, self).__init__()
        
        self.layers = nn.ModuleList()  

        self.layers.append(nn.Linear(inputSize, layerSize))
        
        for i in range(numLayers-1):    
            self.layers.append(nn.Linear(layerSize, layerSize))
            
        self.fcOut = nn.Linear(layerSize, outputSize)



        # initialize weights
        self.initWeights('Xavier')

    def forward(self, x):     
        for layer in self.layers:
            x = F.tanh(layer(x))
  
        x = self.fcOut(x)  # Idendity activation function = No activation
        
        
        return x

    def initWeights(self, method):
        if method == "Xavier":
            # initialize weights and biases

            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight, gain=1)

            nn.init.xavier_normal_(self.fcOut.weight, gain=1)
            
            
            for layer in self.layers:
                nn.init.zeros_(layer.bias)

            nn.init.zeros_(self.fcOut.bias)

        else:
            raise ValueError('Initalizationmethod not yet implemented')



class Hyperparameter(object):
    def __init__(self):
        self.learningRate = None
        self.numEpochs = None
        self.learningRateDecay = None
        self.numEpochsForDecay = None
        self.batchSize = None
        self.optimizer = ''
        self.numLayers = None
        self.numNeuronsPerLayer = None
        self.activation = None
        self.scaling_strategy = None
        self.weight_pde_loss = []
        self.weight_dirichlet_loss = []
        self.weight_neumann_loss = []
        self.weight_robin_loss = []
    
    def get_param_dict(self):
        
        # Save all parameters, that could be used in runs with multiple optimizers as strings
        if  not isinstance(self.learningRate,str):
            self.learningRate = f"{self.learningRate}"
            
        if  not isinstance(self.numEpochs,str):
            self.numEpochs = f"{self.numEpochs}"
                
        if  not isinstance(self.learningRateDecay,str):
            self.learningRateDecay = f"{self.learningRateDecay}"
            
        if  not isinstance(self.numEpochsForDecay,str):
            self.numEpochsForDecay = f"{self.numEpochsForDecay}"
            
        if  not isinstance(self.batchSize,str):
            self.batchlearningRateSize = f"{self.batchSize}"
            
        hyperparmeter_dict = {'lr': self.learningRate, \
                            'epochs': self.numEpochs, \
                            'lr_decay': self.learningRateDecay, \
                            'epochs_for_decay': self.numEpochsForDecay, \
                            'batch_size': self.batchSize, \
                            'optimizer': self.optimizer, \
                            'activation': self.activation, \
                            'num_layers': self.numLayers, \
                            'num_neurons_per_layer': self.numNeuronsPerLayer, \
                            'scaling_strategy': self.scaling_strategy}
            
        return hyperparmeter_dict

        
# training of a ordinary neural network 
def trainNetworkSupervised(model, hyperparameter, X_train, U_train, X_test, U_test):
    if (hyperparameter.optimizer == "Adam"):
        if hyperparameter.batchSize is None:
            print("\n >>> Training model with Adam \n")
            tr_l, val_l = trainNetworkSupervised_Adam(model, hyperparameter, X_train, U_train, X_test, U_test)
            
        else:
            print("\n >>> Training model with Adam with mini batches \n")
            tr_l, val_l = trainNetworkSupervised_Adam_Batches(model, hyperparameter, X_train, U_train, X_test, U_test)

    else:
        raise ValueError("Defined optimizer not yet implemented for ordinary neural network, so far only Adam available")

    return tr_l, val_l
    
            

# training for purely supervised learning
def trainNetworkSupervised_Adam(model, hyperparameter, X_train, U_train, X_test, U_test):

    training_loss = []
    validation_loss = []

    optimizer = Adam(model.parameters(), lr=hyperparameter.learningRate)
    scheduler = StepLR(optimizer, step_size=hyperparameter.numEpochsForDecay, gamma=hyperparameter.learningRateDecay)


    for epoch in range(1, hyperparameter.numEpochs + 1):
        # making a prediction for u
        U_pred = model(X_train)

        # computing the loss
        loss = MSEloss(U_train, U_pred)

        optimizer.zero_grad()

        # propagating backward
        loss.backward()

        # updating parameters
        optimizer.step()

        # update learning rate
        scheduler.step()

        # calculate validation loss
        U_pred = model(X_test)
        loss_valid = MSEloss(U_test, U_pred)

        # save losses in list
        training_loss.append(loss.detach().numpy())
        validation_loss.append(loss_valid.detach().numpy())

        if (epoch == 1 or epoch % 500 == 0):
            print("Epoch %d, train_loss: %.6f, valid_loss: %.6f" % (
                epoch, loss.detach().numpy(), loss_valid.detach().numpy()))

    return training_loss, validation_loss


# training for purely supervised learning in batchmode
def trainNetworkSupervised_Adam_Batches(model, hyperparameter, X_train, U_train, X_test, U_test):

    training_loss = []
    validation_loss = []

    optimizer = Adam(model.parameters(), lr=hyperparameter.learningRate)
    scheduler = StepLR(optimizer, step_size=hyperparameter.numEpochsForDecay, gamma=hyperparameter.learningRateDecay)

    xDim = X_train.shape[1]
    dataSet = torch.cat([X_train, U_train], axis=1)
    data = DataLoader(dataSet, batch_size=hyperparameter.batchSize)

    for epoch in range(1, hyperparameter.numEpochs + 1):
        
        for d in data:
            
            X_train = d[:, 0:xDim]
            U_train = d[:, xDim:]
                
            # making a prediction for u
            U_pred = model(X_train)
    
            # computing the loss
            loss = MSEloss(U_train, U_pred)
    
            optimizer.zero_grad()
    
            # propagating backward
            loss.backward()
    
            # updating parameters
            optimizer.step()
    
            # update learning rate
            scheduler.step()
    
    
        # calculate validation loss
        U_pred = model(X_test)
        loss_valid = MSEloss(U_test, U_pred)
        
        # calculate training loss after epoch
        U_pred = model(X_train)
        loss = MSEloss(U_train, U_pred)

        # save losses in list
        training_loss.append(loss.detach().numpy())
        validation_loss.append(loss_valid.detach().numpy())

        if (epoch == 1 or epoch % 500 == 0):
            print("Epoch %d, train_loss: %.6f, valid_loss: %.6f" % (
                epoch, loss.detach().numpy(), loss_valid.detach().numpy()))

    return training_loss, validation_loss


# testing network
def testNetwork(model, X_test, U_test):
    '''
    X_test and U_test need to be a pytorch tensor 
    '''
    U_predict = model(X_test)
    U_predict = U_predict.detach().numpy()
    U_test = U_test.detach().numpy()
    numOutputs = U_test.shape[0] * U_test.shape[1]
    MSE = np.sum((U_predict-U_test)**2)/numOutputs
    MAE = np.sum(abs((U_predict-U_test)))/numOutputs
    return MSE, MAE






