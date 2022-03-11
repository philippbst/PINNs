#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:53:37 2021

@author: philippbst
"""

def main():

#%% -----------------------------------------------------------------------------------------------
    import os
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from torch.autograd import grad
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib 
    import numpy as np

    import neural_network.network as neu
    import tools.utils as utils
    import visualization.visualization_nn as visnn
    from neural_network.loss_functions import MSEloss
    from tools.pde_information import WaveEquationParams
    from pinn.helmholtz_1D_pinn import Helmholtz1DPINN
    from pinn.utils_pinn import Dataset

#%% -----------------------------------------------------------------------------------------------
    plt.close('all')
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 12}
    matplotlib.rc('font', **font)
    pathDict = utils.getProjectPaths(os.path.dirname(os.path.realpath(__file__))[:-4])


#%% -----------------------  Defining simulation parameter and generating data --------------------

    # sepcify simulation parameter
    x_max = 1
    mode =  3
    c = 1 
    omega = (mode * np.pi * c) / x_max   
    pdeParams = WaveEquationParams(c,omega)

    # Generating data in domain
    numColl = 1000
    x_CP = np.random.rand(numColl).reshape(numColl,1) * x_max

    # Prescribed solution (Dirichlet BC) for mode 3
    x_known = [0,0.4]
    u_known = [0,-0.062365977978723716]

    x_ICBC = np.array(x_known).reshape(len(x_known),1)
    u_ICBC = np.array(u_known).reshape(len(x_known),1)

    X_train_CP = torch.from_numpy(x_CP).float()
    X_train_BC_Dir = torch.from_numpy(x_ICBC).float()
    U_train_BC_Dir = torch.from_numpy(u_ICBC).float()

    # Create dataset for PINN training
    dt = Dataset()
    dt.X_train_CP = X_train_CP
    dt.X_train_BC_Dir = X_train_BC_Dir
    dt.U_train_BC_Dir = U_train_BC_Dir


#%% ---------------------  Hyperparameter definition and training of the PINN  ---------------------
    hp = neu.Hyperparameter()
    hp.learningRate = 0.1
    hp.numEpochs = 200
    hp.learningRateDecay = 0.5
    hp.numEpochsForDecay = 500
    hp.optimizer = "LBFGS"
    hp.numNeuronsPerLayer = 25
    hp.numLayers = 15
    hp.scaling_strategy = 'Balance'

    # creating and training PINN
    model = neu.NeuralNetwork(1, 1, hp.numLayers, hp.numNeuronsPerLayer)
    pinn = Helmholtz1DPINN(pdeParams, model)
    pinn.train_model(hp,dt)

        
#%% --------------------------------  Plotting the learning courves --------------------------------
    pde_loss = r'$\mathcal{L}_{\Omega,i} /\mathcal{L}_{\Omega,1} $'
    bc_loss = r'$\mathcal{L}_{\Gamma,i} /\mathcal{L}_{\Gamma,1} $'

    loss_names = [pde_loss, bc_loss]
    training_losses = np.concatenate((np.array(pinn.training_losses.training_loss_CP).reshape(len(pinn.training_losses.training_loss),1),\
                                        np.array(pinn.training_losses.training_loss_BC).reshape(len(pinn.training_losses.training_loss),1)), axis = 1)

    fig1, ax1 = visnn.plotMultiLearningCurves(training_losses, loss_names, True)



#%% ----------------------------------  testing the network ----------------------------------------
    numTestPoints = 101
    x_test = np.linspace(0,x_max,numTestPoints).reshape([numTestPoints,1])
    U_pred = model(torch.from_numpy(x_test).float()).detach().numpy()



#%% -----------------------  sovle the 1D wave equation numerically --------------------------------

    from scipy.integrate import odeint
    import numpy as np
    import matplotlib.pyplot as plt

    '''
    Governing equation in frequency domain:
    d2u/d2x = -(omega/c)^2 * u

    Transforming second order ODE in system of first order ODEs:
    x1 = u
    x2 = du/dx


    This results in:
    dx1/dx = x2
    dx2/dx = -(omega/c)^2 * x1
    '''

    # set up ODE
    def rhsf(initial,x): 
        # initial values
        x1 = initial[0]
        x2 = initial[1]
            
        # ODE System
        dx1dx = x2
        dx2dx = -(omega/c)**2 * x1
        
        return np.array([dx1dx , dx2dx])

    # define solution space and initial conditions
    x = np.linspace(0,x_max,101)
    BC = np.array([0,1])

    # solve ode
    X = odeint(rhsf, BC, x)
    x1 = X[:,0]
    x2 = X[:,1]

#%% --------------------------  get u(x) at the desired x coordinate -------------------------------
    if False:
        x_des = 0.4
        pos = np.where(x == x_des)[0][0]
        u_des = x1[pos]
        print(u_des)
        
        plt.figure()
        plt.plot(x,x1)
        plt.title('Numerical solution of the 1D Helmoltz equation')
        plt.grid('minor')
        plt.xlabel('x')
        plt.ylabel('u(x)')

#%% ------------------  Plotting the comparison of traditional and PINN solution -------------------
    fig, ax  = plt.subplots()
    ax.plot(x_test,U_pred,'--.', c = 'tab:blue')
    ax.plot(x,x1,'-', c = 'tab:red')
    plt.scatter(x_known,u_known,c = 'k', s = 60)
    ax.legend(['PINN solution','True solution','Given data points'], loc='upper right')
    ax.grid('minor')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p(x)$')
    csfont = {'fontname': 'Arial'}
    ax.set_title('PINN and true solution' ,**csfont)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.show()

#%% ------------  Calculating mean absolute deviation of PINN and trafitional solution -------------
    MAE = np.sum(abs(abs(x1.reshape(np.size(x1),1))  - abs((U_pred)))) / np.shape(U_pred)[0]
    U_mean = np.sum(abs(x1.reshape(np.size(x1),1))) / np.shape(U_pred)[0]

    error = (MAE / U_mean) * 100

    print('')
    print('Mean absolute deviation is: ' + str(round(MAE,8)))
    print('Mean deviation of predictions: ' + str(round(error,2)) + '[%]')
    print('')


#%% ------------------------------------  Save or Load model ---------------------------------------
    save_model  = False
    if save_model:
        folderName = "1Dhelmholtz"
        modelName = '1D_Helmholtz_3Mode_15Layers_25NeuPerLayer_XXX'
        savePath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
        torch.save(pinn.model.state_dict(), savePath)


    load_model = True
    if load_model:
        folderName = "1Dhelmholtz"
        modelName = '1D_Helmholtz_3Mode_15Layers_25NeuPerLayer'
        x_known = [0,0.4] # used data for training
        u_known = [0,-0.062365977978723716] # used data for training
        
        model = neu.NeuralNetwork(1, 1, 15, 25)
        loadPath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
        model.load_state_dict(torch.load(loadPath))
        mode = 3

#%% -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()



# %%
