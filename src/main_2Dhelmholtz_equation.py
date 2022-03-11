#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:17:14 2021

@author: philippbst
"""

def main():
#%% -----------------------------------------------------------------------------------------------
    import os
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import ticker
    import numpy as np

    import neural_network.network as neu
    import visualization.visualization_nn as visnn
    import tools.utils as utils
    from tools.pde_information import WaveEquationParams
    from pinn.helmholtz_2D_pinn import Helmholtz2DPINN
    from pinn.utils_pinn import Dataset


#%% ----------------------- Select scope and settings of the script --------------------------------
    trainPhysicsInformed = True
    trainSupervised = False

    # Select what data points to use for Dirichlet BC
    onlyBoundaryPointsAndOneLine = False
    pointGridInDomainAndBoundray = False
    sparseDataInDomainAndBoundary = True


#%% -----------------------------------------------------------------------------------------------
    plt.close('all')
    pathDict = utils.getProjectPaths(os.path.dirname(os.path.realpath(__file__))[:-4])


#%% --------------  True solution of the 2D Helmoltz equation for a squared membrane ---------------
    def helmholtz2D_Solution(x,y,mode,x_max):
        u = np.sin((mode * np.pi * x)/x_max) * np.sin((mode * np.pi * y)/x_max)
        return u


#%% -----------------------  Defining simulation parameter and generating data --------------------
    # Domain information
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = x_max

    # PDE information
    mode = 2
    c = 1
    omega = c * np.pi * (((mode/x_max)**2+(mode/y_max)**2)**0.5)
    pdeParams = WaveEquationParams(c,omega)


    # Random collocation points in the domain 
    numColl = 2000
    x = np.random.rand(numColl).reshape(numColl,1) * x_max
    y = np.random.rand(numColl).reshape(numColl,1) * y_max
    X_CP = np.concatenate((x,y), axis = 1)


    if onlyBoundaryPointsAndOneLine:
        # Only using Dirichlet BC on the boundary
        # BC points
        numBCperBoundary = 100
        
        # Dirichlet BC at boundary x1
        y_bc_x1 = np.random.rand(numBCperBoundary).reshape(numBCperBoundary,1) * y_max
        x_bc_x1 = np.ones_like(y_bc_x1) * x_min
        X_bc_x1 = np.concatenate((x_bc_x1,y_bc_x1), axis = 1)
        U_bc_x1 = np.zeros_like(y_bc_x1) 
        
        '''
        # Dirichlet BC at boundary x2
        y_bc_x2 = np.random.rand(numBCperBoundary).reshape(numBCperBoundary,1) * y_max
        x_bc_x2 = np.ones_like(y_bc_x2) * x_max
        X_bc_x2 = np.concatenate((x_bc_x2,y_bc_x2), axis = 1)
        U_bc_x2 = np.zeros_like(y_bc_x2) 
        '''
        
        #''' for defining the boundary at the middle of the membrane
        y_bc_x2 = np.random.rand(numBCperBoundary).reshape(numBCperBoundary,1) * y_max
        x_bc_x2 = np.ones_like(y_bc_x2) * x_max*0.5
        X_bc_x2 = np.concatenate((x_bc_x2,y_bc_x2), axis = 1)
        U_bc_x2 = helmholtz2D_Solution(x_bc_x2,y_bc_x2,mode,x_max)

        # Dirichlet BC at boundary y1
        x_bc_y1 = np.random.rand(numBCperBoundary).reshape(numBCperBoundary,1) * x_max
        y_bc_y1 = np.ones_like(x_bc_y1) * x_min
        X_bc_y1 = np.concatenate((x_bc_y1,y_bc_y1), axis = 1)
        U_bc_y1 = np.zeros_like(x_bc_y1) 
        
        # Dirichlet BC at boundary y1
        x_bc_y2 = np.random.rand(numBCperBoundary).reshape(numBCperBoundary,1) * x_max
        y_bc_y2 = np.ones_like(x_bc_y2) * x_max
        X_bc_y2 = np.concatenate((x_bc_y2,y_bc_y2), axis = 1)
        U_bc_y2 = np.zeros_like(x_bc_y2) 
        
        # merging all boundary conditions in one vector
        X_BC = np.concatenate((X_bc_x1,X_bc_x2,X_bc_y1,X_bc_y2), axis = 0)
        U_BC = np.concatenate((U_bc_x1,U_bc_x2,U_bc_y1,U_bc_y2), axis = 0)

    if pointGridInDomainAndBoundray:
        # Using solution points in the domain as BC points, some of them are also on the boundary and thus dirichlet BC (depends on choics of points)
        
        # creating data
        numPointsPerLine = 20
        boundaryOffset = 0
        x = np.linspace(x_min+boundaryOffset,x_max-boundaryOffset,numPointsPerLine)
        y = np.linspace(y_min+boundaryOffset,y_max-boundaryOffset,numPointsPerLine)
        xx,yy = np.meshgrid(x,y)
        
        x_linearized = xx.flatten()
        y_linearized = yy.flatten()
        x_linearized = x_linearized.reshape(x_linearized.shape[0],1)
        y_linearized = y_linearized.reshape(y_linearized.shape[0],1)
        
        U_BC = helmholtz2D_Solution(xx,yy,mode,x_max)
        U_BC = U_BC.reshape(x_linearized.shape)
        X_BC = np.concatenate((x_linearized,y_linearized), axis = 1)
    
    if sparseDataInDomainAndBoundary:
        # More controlled sampling points in the domain and on the boundary
        
        # sampling points on each domain boundary
        numPointsPerLine = 8
        boundaryOffset = 0
        bl = np.linspace(x_min+boundaryOffset,x_max-boundaryOffset,numPointsPerLine).reshape(numPointsPerLine,1)

        # Dirichlet BC at boundary x1
        y_bc_x1 = bl
        x_bc_x1 = np.ones_like(y_bc_x1) * x_min
        X_bc_x1 = np.concatenate((x_bc_x1,y_bc_x1), axis = 1)
        U_bc_x1 = np.zeros_like(y_bc_x1) 
        
        # Dirichlet BC at boundary x2
        y_bc_x2 = bl
        x_bc_x2 = np.ones_like(y_bc_x2) * x_max
        X_bc_x2 = np.concatenate((x_bc_x2,y_bc_x2), axis = 1)
        U_bc_x2 = np.zeros_like(y_bc_x2) 

        
        # Dirichlet BC at boundary y1
        x_bc_y1 = bl
        y_bc_y1 = np.ones_like(x_bc_y1) * x_min
        X_bc_y1 = np.concatenate((x_bc_y1,y_bc_y1), axis = 1)
        U_bc_y1 = np.zeros_like(x_bc_y1) 
        
        # Dirichlet BC at boundary y1
        x_bc_y2 = bl
        y_bc_y2 = np.ones_like(x_bc_y2) * x_max
        X_bc_y2 = np.concatenate((x_bc_y2,y_bc_y2), axis = 1)
        U_bc_y2 = np.zeros_like(x_bc_y2) 
        
        # merging all boundary conditions in one vector
        X_BC_DB = np.concatenate((X_bc_x1,X_bc_x2,X_bc_y1,X_bc_y2), axis = 0)
        U_BC_DB = np.concatenate((U_bc_x1,U_bc_x2,U_bc_y1,U_bc_y2), axis = 0)
        
        
        # sampling sorted points in the domain 
        numPointsPerLine = 2
        boundaryOffset = 0.4
        x = np.linspace(x_min+boundaryOffset,x_max-boundaryOffset,numPointsPerLine).reshape(numPointsPerLine,1)
        y = np.linspace(x_min+boundaryOffset,x_max-boundaryOffset,numPointsPerLine).reshape(numPointsPerLine,1)
        xx,yy = np.meshgrid(x,y)
        x_linearized = xx.flatten()
        y_linearized = yy.flatten()
        x = x_linearized.reshape(x_linearized.shape[0],1)
        y = y_linearized.reshape(y_linearized.shape[0],1)
        
        X_BC_D = np.concatenate((x,y), axis = 1)
        U_BC_D = helmholtz2D_Solution(x,y,mode,x_max)
        
        X_BC =  np.concatenate((X_BC_DB,X_BC_D), axis = 0)
        U_BC = np.concatenate((U_BC_DB,U_BC_D), axis = 0)

    # creating tensors and dataset for PINN training  
    X_train_CP = torch.from_numpy(X_CP).float()
    X_train_ICBC = torch.from_numpy(X_BC).float()
    U_train_ICBC = torch.from_numpy(U_BC).float()

    dt = Dataset()
    dt.X_train_CP = X_train_CP
    dt.X_train_BC_Dir = X_train_ICBC
    dt.U_train_BC_Dir = U_train_ICBC

    
#%% -------------------------------  Plot the data used for training -------------------------------
    fig, ax = plt.subplots()
    p = ax.scatter(X_BC[:,0], X_BC[:,1], c = 'tab:blue');
    ax.scatter(X_CP[:,0],X_CP[:,1], c = 'tab:green', alpha = 0.2)

    ax.plot((0,0), (0,1), 'k-')
    ax.plot((0,1), (0,0), 'k-')
    ax.plot((1,0), (1,1), 'k-')
    ax.plot((1,1), (1,0), 'k-')
    ax.set_title('Data points used for training')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')


#%% ------------------------------  Setting up the model and training ------------------------------
    hp = neu.Hyperparameter()

    if trainPhysicsInformed:
        # define hyperparameter
        hp.learningRate = 0.1
        hp.numEpochs = 500
        hp.optimizer = 'LBFGS'
        hp.learningRateDecay = 0.5
        hp.numEpochsForDecay = 1500
        hp.numNeuronsPerLayer = 10
        hp.numLayers = 5
        hp.scaling_strategy = 'Balance'
        
        # create and train model
        model = neu.NeuralNetwork(2, 1, hp.numLayers, hp.numNeuronsPerLayer)
        pinn = Helmholtz2DPINN(pdeParams, model)
        pinn.train_model(hp,dt)
        
        # plotting the loss courves
        pde_loss = r'$\mathcal{L}_{\Omega,i} /\mathcal{L}_{\Omega,1} $'
        bc_loss = r'$\mathcal{L}_{\Gamma,i} /\mathcal{L}_{\Gamma,1} $'
        loss_names = [pde_loss, bc_loss]
        training_losses = np.concatenate((np.array(pinn.training_losses.training_loss_CP).reshape(len(pinn.training_losses.training_loss),1),\
                                        np.array(pinn.training_losses.training_loss_BC).reshape(len(pinn.training_losses.training_loss),1)), axis = 1)
        
        fig1, ax1 = visnn.plotMultiLearningCurves(training_losses, loss_names, True)
        
    if trainSupervised:
        # defining hyperparameter
        hp.learningRate = 0.01
        hp.numEpochs = 5000
        hp.learningRateDecay = 0.5
        hp.numEpochsForDecay = 1000
        hp.numNeuronsPerLayer = 10
        hp.numLayers = 5
        
        # create and train model
        model = neu.NeuralNetwork(2, 1, hp.numLayers, hp.numNeuronsPerLayer)
        training_loss,_ = neu.trainNetworkSupervised(model,hp,X_train_ICBC,U_train_ICBC,X_train_ICBC,U_train_ICBC)
        
        # plotting the loss courves
        loss_names = ['Training loss']
        training_loss = np.array(training_loss).reshape(len(training_loss),1)
        fig1, ax1 = visnn.plotMultiLearningCurves(training_loss, loss_names, True)



#%% ----------------------------  testing the network and plotting results -------------------------

    # creating testing data
    stepSize = 0.01
    x = np.arange(x_min,x_max,stepSize)
    y = np.arange(y_min,y_max,stepSize)
    xx,yy = np.meshgrid(x,y)

    x_linearized = xx.flatten()
    y_linearized = yy.flatten()
    x_linearized = x_linearized.reshape(x_linearized.shape[0],1)
    y_linearized = y_linearized.reshape(y_linearized.shape[0],1)

    X_test = np.concatenate((x_linearized,y_linearized), axis = 1)
    X_test_tensor = torch.from_numpy(X_test).float()

    U_pred = model(X_test_tensor).detach().numpy()

    # plot PINN predictions
    U_pred_mesh = U_pred.reshape(x.shape[0],y.shape[0])
    fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, U_pred_mesh, cmap=cm.RdBu,linewidth=1, antialiased=True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zticks([])
    cbar = fig2.colorbar(surf, shrink=0.6, aspect=8)
    cbar.ax.set_title(r'$\hat{p}$')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax.set_title('Prediction of the PINN')


    # Plot analytical solution 
    u_true = helmholtz2D_Solution(xx,yy,mode,x_max)
    fig3, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, u_true, cmap=cm.RdBu,linewidth=1, antialiased=True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zticks([])
    cbar = fig3.colorbar(surf, shrink=0.6, aspect=8)
    cbar.ax.set_title(r'$p$')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax.set_title('Analytical solution')


    # Plot deviation of true and PINN solution
    dev = u_true - U_pred_mesh 
    fig4, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, dev, cmap=cm.gist_heat,linewidth=1, antialiased=True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zticks([])
    cbar = fig4.colorbar(surf, shrink=0.6, aspect=8)
    cbar.ax.set_title(r'$p-\hat{p}$')
    ax.set_title('Deviation of analytical and PINN solution')

    # Calculate error measures
    U_mean = np.sum(abs(u_true)) / (u_true.shape[0]*u_true.shape[0])
    Mean_deviation = np.sum(abs(u_true - U_pred_mesh)) / (u_true.shape[0]*u_true.shape[0])
    error = (Mean_deviation / U_mean) * 100
    print('')
    print('----------------- Performance --------------------')
    print('Mean absolute deviation is: ' + str(round(Mean_deviation,8)))
    print('Mean deviation of predictions: ' + str(round(error,2)) + '[%]')
    print('')

#%% -----------------  Plots of 1D Crossection crossing the peaks of second mode -------------------
    x = np.arange(0,x_max,stepSize).reshape(np.arange(0,x_max,stepSize).size,1)
    y = 0.25
    u_true_1D = helmholtz2D_Solution(x,y,mode,x_max)

    y_model = np.ones_like(x) * y
    X_test_1D = torch.from_numpy(np.concatenate((x, y_model), axis = 1)).float()
    u_pred_1D_pinn = pinn.model(X_test_1D).detach().numpy()
    u_pred_1D = model(X_test_1D).detach().numpy()

    fig, ax  = plt.subplots()
    ax.plot(x,u_pred_1D_pinn,'--.', c = 'tab:blue')
    ax.plot(x,u_true_1D,'-', c = 'tab:red')
    ax.legend(['PINN solution','Ordinary NN solution','True solution'])
    ax.grid('minor')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(rf'$u(x, y = {y})$')
    ax.set_title('Comparison between PINN and ordinary NN')

    plt.show()

#%% --------------------------------------  Save and load model ------------------------------------
    save_model  = False
    if save_model:
        folderName = "2Dhelmholtz"
        modelName = '2D_Helmholtz_2Mode_5Layers_10NeuPerLayer'
        savePath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
        torch.save(pinn.model.state_dict(), savePath)


    load_model = False
    if load_model:
        folderName = "2Dhelmholtz"
        # First trained model 1DHwithout optimization for High Excitation 
        modelName = '2D_Helmholtz_2Mode_5Layers_10NeuPerLayer'
        
        model = neu.NeuralNetwork(2, 1, 5, 10)
        loadPath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
        model.load_state_dict(torch.load(loadPath))
        pinn = Helmholtz2DPINN(None, model)


#%% -------------------------------------  Show Robin BC -------------------------------------------
# Dirichlet BC at boundary x1
    y = np.linspace(0,1,5).reshape(5,1)
    x = np.ones_like(y) * 0
    X = np.concatenate((x,y), axis = 1)

    N = np.ones_like(X) * -1
    N[:,1] = 0

    Y = np.ones_like(x)

    X = torch.from_numpy(X).float()
    N = torch.from_numpy(N).float()
    Y = torch.from_numpy(Y).float()

    lossBC = pinn.robinBC(X, Y, N)

#%% -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()



# %%