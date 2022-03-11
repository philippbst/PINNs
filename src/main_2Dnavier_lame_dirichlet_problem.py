#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:11:46 2021

@author: philippbst
"""


def main():
#%% -----------------------------------------------------------------------------------------------    
    import time
    import os
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    from sklearn.model_selection import train_test_split
    import torch
    from ray import tune
    from scipy import interpolate

    from tools.pde_information import NavierLameParams
    import neural_network.network as neu
    from pinn.navier_lame_2D_pinn import NavierLame2DPINN
    import tools.comsol_data_processor as cdp
    import tools.utils as utils
    import visualization.visualization_nn as visnn
    from visualization.visualization_general import plot_interpolated_squared_field
    import visualization.visualization_general as visgen
    from pinn.utils_pinn import ProgressTracker, TensorboardWriterAssistant, RayTuneSetup, Dataset

#%% ----------------------------------------- Settings ----------------------------------------------

    fileName = "2DNavierLame_Displacement_MeshNormal.txt"

    # Define the used solution and the desired training method
    FREQUENCY = 2500 
    AMPLFICATION = 1

    # scaling FEM solution before training and PINN predictions
    scalingFEMData = 0.1
    scalingPredictions = 1

    # Select training type
    trainSupervised = False
    trainPhysicsInformed = True

    # select what FEM data for training
    onlyBoundaryPoints = True # in this case testing data is whole FEM dataset in domain and on boundary
    allFEMPoints = False # in this case the FEM points are splitted into 70% training, 30% testing data

    # select what to visualize
    showFEMData = True
    showTrainingData = True

    # Save data as tensorboard run
    save_tensorboard_run = False

    # ray tune study 
    ray_tune_study = False

    # use progress tracker for training
    trackTrainingProcess = False


#%%  -------------------------------  Generating / collecting data ---------------------------------

    # get Paths
    pathDict = utils.getProjectPaths(os.path.dirname(os.path.realpath(__file__))[:-4])
    data = cdp.loadComsolDisplacements(pathDict, fileName)

    # Create Comsol data object and extract and process raw data
    SimulationData = cdp.ComsolData(data)
    SimulationData.ExtractInformation()

    nodeKoord = SimulationData.nodeKoord
    nodeDisplacements = SimulationData.nodeDisplacements
    freqRange = SimulationData.frequencyRange

    X = nodeKoord
    U = nodeDisplacements[FREQUENCY] * scalingFEMData

    # Ignore the third dimension
    U = U[:, :2]


    # Plot FEM data with deformation
    if showFEMData:
        vis_FEM = visgen.Visualizer(X,U, SimulationData.dimension)
        title = 'FEM nodes with deformation'
        _, ax1 = vis_FEM.plot_with_deformation(AMPLFICATION, title)
        
        # Show the undeformed geometry in the plot
        ax1.plot((-0.5, 0.5), (0.25, 0.25), 'k-')
        ax1.plot((-0.5, 0.5), (-0.25, -0.25), 'k-')
        ax1.plot((0.5, 0.5), (-0.25, 0.25), 'k-')
        ax1.plot((-0.5, -0.5), (-0.25, 0.25), 'k-')
        plt.show()


    # prepare the data from the FEM simulation
    if onlyBoundaryPoints:
        idx_boun = np.where((abs(X[:, 0]) == 0.5) |  (abs(X[:, 1]) == 0.25))[0]
        X_train = X[idx_boun, :]
        U_train = U[idx_boun, :]
        X_test = X
        U_test = U  
    elif allFEMPoints:
        X_train, X_test, U_train, U_test = train_test_split(X, U, test_size=0.3)

    # create collocation points sampler
    def CPSampler_2D(NUMBER_COLLOCATION_POINTS):
        # sample points in the domain and on the boundary in [m]
        x_range = [-0.5, 0.5]
        y_range = [-0.25, 0.25]
        z_range = [0, 0]

        # Collocation points points in the domain
        X_coll = utils.pointsInRange3D(NUMBER_COLLOCATION_POINTS, x_range, y_range, z_range)
        X_coll = X_coll[:, :2]
        X_train_CP = torch.from_numpy(X_coll).float()
        return X_train_CP

    NUMBER_COLLOCATION_POINTS = 2000
    X_train_CP = CPSampler_2D(NUMBER_COLLOCATION_POINTS)

    # plot training data 
    if showTrainingData:
        fig, ax = plt.subplots()
        ax.scatter(X_train[:,0], X_train[:,1], c = 'tab:blue')
        ax.scatter(X_train_CP[:,0],X_train_CP[:,1], c = 'tab:green', alpha = 0.2)
        ax.set_title('Data points used for training')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.plot((-0.5,-0.5), (-0.25,0.25), 'k-')
        ax.plot((-0.5,0.5), (-0.25,-0.25), 'k-')
        ax.plot((0.5,-0.5), (0.25,0.25), 'k-')
        ax.plot((0.5,0.5), (0.25,-0.25), 'k-')
        ax.set_aspect('equal')
        plt.show()
        


#%% ------------------- Defining PDE parameter and create dataset for PINN --------------------------
    E = 205e9                   # in [Pa = N/m^2]
    nu = 0.28
    lam = (nu/(1-2*nu)) * (1/(1+nu)) * E
    my = 1/2 * (1/(1+nu)) * E
    LAMBDA = lam                # in [N/m^2]
    MY = my                     # in [N/m^2]
    OMEGA = 2*np.pi*FREQUENCY   # 2*pi*FREQUENCY
    RHO = 7850                  # in [kg / m^3]

    pdeParams = NavierLameParams(LAMBDA, MY, RHO, OMEGA)

   # transform into tensors
    X_train = torch.from_numpy(X_train).float()
    U_train = torch.from_numpy(U_train).float()

    X_test = torch.from_numpy(X_test).float()
    U_test = torch.from_numpy(U_test).float()

    dt = Dataset()
    dt.X_train_CP = X_train_CP
    dt.X_train_BC_Dir = X_train
    dt.U_train_BC_Dir = U_train
    dt.X_test = X_test
    dt.U_test = U_test


#%% ---------------------- Create progress tracker and tensorboard writer -----------------------------
    if trackTrainingProcess:
        epochSteps = 5 
        tracker = ProgressTracker(epochSteps, X)
    else:
        tracker = None

    # create writer for tensorboard
    if save_tensorboard_run:
        path_to_runs = pathDict.get('pathToModelRuns') + "/2Dnavier_lame"
        w = TensorboardWriterAssistant(path_to_runs)
        w.create_writer()
    else:
        w = None



#%% ------------------------------  Setting up the model and training ------------------------------
    inputSize = 2
    outputSize = 2

    if ray_tune_study and trainPhysicsInformed:
        
        config = {
            "learningRate": tune.grid_search([0.1, 0.0001]),
            "numEpochs": tune.grid_search([10, 20]),
            "optimizer": tune.choice(["Adam"]),
            "learningRateDecay": tune.grid_search([0.3]),
            "numEpochsForDecay": tune.grid_search([500]),
            "batchSize": tune.grid_search([None]),
            "numNeuronsPerLayer": tune.grid_search([5]),
            "numLayers": tune.grid_search([5]),
            "scaling_strategy": tune.choice(["Balance"]),
        }  
        
                
        def training_function(config):
            
            # wait random time between 0-10 seconds to make sure all run folders are written 
            rand = np.random.rand()*10
            time.sleep(rand)
            hp = neu.Hyperparameter()

            # parameters to be involved in grid search
            hp.learningRate = config.get("learningRate")
            hp.numEpochs = config.get("numEpochs")
            hp.optimizer = config.get("optimizer")
            hp.learningRateDecay = config.get("learningRateDecay")
            hp.numEpochsForDecay = config.get("numEpochsForDecay")
            hp.batchSize = config.get("batchSize")
            hp.numNeuronsPerLayer = config.get("numNeuronsPerLayer")
            hp.numLayers = config.get("numLayers")
            hp.scaling_strategy = config.get("scaling_strategy")
            

            # model creation 
            model = neu.NeuralNetwork(inputSize, outputSize, hp.numLayers, hp.numNeuronsPerLayer)

            #creating tensorboard writer
            w = TensorboardWriterAssistant(path_to_study_runs)
            w.create_writer()
                
            # creating and training Pinn
            pinn = NavierLame2DPINN(pdeParams, model, CPSampler_2D, None, w, True)
            pinn.train_model(hp, dt)
            
            # adding stuff to tensorboard
            hyperparameter_dict = hp.get_param_dict()
            metric_dict = pinn.get_final_metric_dict()
            w.writer.add_hparams(hyperparameter_dict, metric_dict, run_name = "res")

            # close writer        
            w.writer.close()
            
            

        path_to_studies = pathDict.get('pathToModelStudies') + "/2Dnavier_lame" 

        tuner = RayTuneSetup(training_function, config, path_to_studies)

        num = tuner.get_next_study_number()
        path_to_study_runs = f"{path_to_studies}/study_{num}/runs"

        max_num_CPUs = 4 
        analysis = tuner.execute_analysis(max_num_CPUs)
        df = analysis.dataframe()
        
        # save dataframe with results to study folder
        df_name = "result_df.pkl"
        path = f"{path_to_studies}/study_{num}/{df_name}"
        df.to_pickle(path)
    
        

    elif (not ray_tune_study and trainSupervised):

            hp = neu.Hyperparameter()
            hp.optimizer = 'Adam'
            hp.learningRate = 0.02
            hp.numEpochs = 6000
            hp.learningRateDecay = 0.5
            hp.numEpochsForDecay = 1000
            hp.batchSize = None
            hp.numNeuronsPerLayer = 25
            hp.numLayers = 5
        
            model = neu.NeuralNetwork(inputSize, outputSize, hp.numLayers, hp.numNeuronsPerLayer)
            training_loss, validation_loss = neu.trainNetworkSupervised(model, hp, X_train, U_train, X_test, U_test)
            loss_names = ['Training loss', 'Validation loss']
            training_losses = np.concatenate((np.array(training_loss).reshape(len(training_loss), 1),
                                            np.array(validation_loss).reshape(len(validation_loss), 1)), axis=1)
        
            fig1, ax1 = visnn.plotMultiLearningCurves(training_losses, loss_names, True)
        
 
    elif (not ray_tune_study and trainPhysicsInformed):

            hp = neu.Hyperparameter()
            hp.optimizer = 'Adam'
            hp.learningRate = 0.005
            hp.numEpochs = 2000
            hp.learningRateDecay = 0.3
            hp.numEpochsForDecay = 500
            hp.batchSize = 5
            hp.numNeuronsPerLayer = 10
            hp.numLayers = 5
            hp.scaling_strategy = 'Balance'

            # creating and training model
            model = neu.NeuralNetwork(inputSize, outputSize, hp.numLayers, hp.numNeuronsPerLayer)
            pinn = NavierLame2DPINN(pdeParams, model, CPSampler_2D, tracker, w)
            pinn.train_model(hp, dt)
            
            # # add further trianing with other optimizer
            # hp.learningRate = 0.1
            # hp.numEpochs = 5 #100
            # hp.learningRateDecay = 0.5
            # hp.numEpochsForDecay = 10000
            # hp.batchSize = None
            # hp.optimizer = 'LBFGS'
            # pinn.train_model(hp, dt)

            # extracting and plotting loss courves
            training_losses = np.concatenate((np.array(pinn.training_losses.training_loss_CP).reshape(len(pinn.training_losses.training_loss), 1),
                                            np.array(pinn.training_losses.training_loss_BC).reshape(len(pinn.training_losses.training_loss), 1)), axis=1)
        
            pde_loss = r'$\mathcal{L}_{\Omega,i} /\mathcal{L}_{\Omega,1} $'
            bc_loss_dir = r'$\mathcal{L}_{\Gamma_{u},i} /\mathcal{L}_{\Gamma_{u},1} $'
            
            loss_names = [pde_loss, bc_loss_dir]
    
            fig1, ax1 = visnn.plotMultiLearningCurves(training_losses, loss_names, True)
            
            
#%% ------------------ show training progress and finish tensorboard run safe ------------------------
    if not ray_tune_study: 

        if trackTrainingProcess:
            maxNumOfPlots = 6
            tracker.visualizePredictions(maxNumOfPlots)
            tracker.visualizeDeviations(maxNumOfPlots, U)
            plt.show()


        if save_tensorboard_run:    
            
            # adding stuff to tensorboard
            hyperparameter_dict = hp.get_param_dict()
            metric_dict = pinn.get_final_metric_dict()
            w.writer.add_hparams(hyperparameter_dict, metric_dict, run_name = "res")
            
            # close writer        
            w.writer.close()


#%% --------------------------------------  Save and load model ------------------------------------
        save_model  = False
        if save_model:
            folderName = "2Dnavier_lame"
            modelName = '2D_MeshNormal_OnlyBoundary_2500Hz_5Layers_25NeuPerLayer_SupervisedOnlyXXXX'
            savePath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
            torch.save(model.state_dict(), savePath)
        
        
        load_model = True
        if load_model:
            folderName = "2Dnavier_lame"
            
            modelName = '2D_MeshNormal_OnlyBoundary_2500Hz_5Layers_25NeuPerLayer_Study3_Run219_Refined'
            #modelName = '2D_MeshNormal_OnlyBoundary_2500Hz_5Layers_25NeuPerLayer_SupervisedOnly'
            loadPath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
                        
            model = neu.NeuralNetwork(2, 2, 5, 25)
            model.load_state_dict(torch.load(loadPath))
            
            scalingPredictions = 0.1

            
            
#%% ----------------------------  testing the network and plotting results -------------------------
    from timeit import default_timer as timer


    # make prediction
    start = timer()
    U_pred = model(torch.from_numpy(X).float()).detach().numpy() * scalingPredictions
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282

#%%
    # plot FEM solution
    field_name = r'$ \parallel\mathbf{u}\parallel_{2}$' + ' (m)'
    U_ges = ((U[:,0])**2 + (U[:,1])**2)**0.5
    tit = 'FEM solution'
    fig, ax = plot_interpolated_squared_field(X, U_ges, levels = 100, field_name = field_name)
    ax.set_xlabel(r'$x$' + ' (m)')
    ax.set_ylabel(r'$y$' + ' (m)')
    csfont = {'fontname':'Arial'}
    plt.title(tit,**csfont)

    # plot PINN prediction
    field_name = r'$ \parallel\hat{\mathbf{u}}\parallel_{2}$' + ' (m)'
    U_pred_ges = ((U_pred[:,0])**2 + (U_pred[:,1])**2)**0.5
    tit = 'Solution predicted by the PINN'
    fig, ax = plot_interpolated_squared_field(X, U_pred_ges, levels = 100, field_name = field_name, \
                                            norm_cbar = True, X_norm = X, U_norm = U_ges)
    ax.set_xlabel(r'$x$' + ' (m)')
    ax.set_ylabel(r'$y$' + ' (m)')
    csfont = {'fontname':'Arial'}
    plt.title(tit,**csfont)

    # plot deviation between FEM and PINN solution
    field_name = r'$| \parallel\mathbf{u}\parallel_{2} - \parallel\hat{\mathbf{u}}\parallel_{2}$' + ' (m)'
    dev = abs(U_pred_ges - U_ges)
    tit = 'Deviation'
    fig, ax = plot_interpolated_squared_field(X, dev, levels = 100, cmap = cm.gist_heat, field_name = field_name)
    ax.set_xlabel(r'$x$' + ' (m)')
    ax.set_ylabel(r'$y$' + ' (m)')
    csfont = {'fontname':'Arial'}
    plt.title(tit,**csfont)


    #Plots of 1D line 
    stepSize = 0.01

    const_koord_min_1D = -0.01
    const_koord_max_1D = 0.01

    X_cross_1D_idx = np.where((X[:,1] > const_koord_min_1D) & (X[:,1] < const_koord_max_1D))[0]
    X_cross_1D = X[X_cross_1D_idx]

    u_cross_1D_disp = U_ges[X_cross_1D_idx]
    u_cross_1D_pred_disp = U_pred_ges[X_cross_1D_idx]

    y_old = X_cross_1D[:,0]
    y_new = np.arange(-0.5,0.5,stepSize)

    f_pred = interpolate.interp1d(y_old,u_cross_1D_pred_disp, kind = 'quadratic')
    u_pred_interp = f_pred(y_new)

    f_true = interpolate.interp1d(y_old,u_cross_1D_disp, kind = 'quadratic')
    u_true_interp = f_true(y_new)

    fig, ax  = plt.subplots()
    ax.plot(y_new,u_pred_interp,'--.', c = 'tab:blue')
    ax.plot(y_new,u_true_interp,'-', c = 'tab:red')
    ax.legend(['Neural network solution','FEM solution'])
    ax.grid('minor')
    ax.set_xlabel(r'$x$' + ' (m)')
    ax.set_ylabel(r'$\parallel\mathbf{u}\parallel_{2}(x, y = 0)$'+ ' (m)')
    ax.set_title('Comparison between NN and FEM solution')

    plt.show()

#%% ------------------------------ MAE on boundary and in the domain --------------------------------

    # MAE in the domain
    idx_dom = np.where((abs(X[:, 0]) != 0.5) & (abs(X[:, 1]) != 0.25))[0]
    X_dom = X[idx_dom, :]
    U_dom = U[idx_dom, :]

    # Calculate measures
    MAE_func = torch.nn.L1Loss()
    MSE_func = torch.nn.MSELoss()

    U_pred = model(torch.from_numpy(X_dom).float())  * scalingPredictions
    MAE = MAE_func(U_pred, torch.from_numpy(U_dom).float()).detach().numpy()
    MSE = MSE_func(U_pred, torch.from_numpy(U_dom).float()).detach().numpy()

    # Mean displacements in the dataset
    U_mean = (np.mean(abs(U_dom[:,0]))+np.mean(abs(U_dom[:,1])))/2

    # Mean deviation of NN predictions in domain compared to FEM solution
    dev = MAE/U_mean * 100

    print('')
    print('--------- Performance for FEM points in domain -------------')
    print('Mean absolute error of predictions: ' + str(MAE) + '[m]')
    print('Mean squared error of predictions: ' + str(MSE) + '[m]')
    print('Mean deviation of predictions: ' + str(round(dev,2)) + '[%]')
    print('')
        
    # MAE for boundary
    idx_boun = np.where((abs(X[:, 0]) == 0.5) |  (abs(X[:, 1]) == 0.25))[0]
    X_boun = X[idx_boun, :]
    U_boun = U[idx_boun, :]

    # Calculate measures
    MAE_func = torch.nn.L1Loss()
    MSE_func = torch.nn.MSELoss()
    
    U_pred = model(torch.from_numpy(X_boun).float()) * scalingPredictions 
    MAE = MAE_func(U_pred, torch.from_numpy(U_boun).float()).detach().numpy()
    MSE = MSE_func(U_pred, torch.from_numpy(U_boun).float()).detach().numpy()
    
    # Mean displacements in the dataset
    U_mean = (np.mean(abs(U_boun[:,0]))+np.mean(abs(U_boun[:,1])))/2
    
    # Mean deviation of NN predictions in domain compared to FEM solution
    dev = MAE/U_mean * 100
    
    print('')
    print('---- Performance for FEM points on boundary (training data)----')
    print('Mean absolute error of predictions: ' + str(MAE) + '[m]')
    print('Mean squared error of predictions: ' + str(MSE) + '[m]')
    print('Mean deviation of predictions: ' + str(round(dev,2)) + '[%]')
    print('')
        

#%% -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()


# %%
