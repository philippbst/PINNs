#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 09:21:06 2021

@author: philippbst
"""

def main():
#%% -----------------------------------------------------------------------------------------------    
    import os
    import time
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    import torch
    from ray import tune
    from scipy import interpolate

    from tools.pde_information import NavierLameParams
    import neural_network.network as neu
    from pinn.navier_lame_3D_pinn import NavierLame3DPINN
    import tools.comsol_data_processor as cdp
    import tools.utils as utils
    import visualization.visualization_nn as visnn
    import visualization.visualization_general as visgen
    from pinn.utils_pinn import TensorboardWriterAssistant, RayTuneSetup, Dataset


#%% ----------------------------------------- Settings ----------------------------------------------

    fileName = '3DNavierLame_Ex2_Displacement_MeshFine_ExcitLow.txt'

    # Define the used solution and the desired training method
    FREQUENCY = 2180
    AMPLFICATION = 1

    # scaling FEM solution before training and PINN predictions
    scalingFEMData = 1
    scalingPredictions = 1

    # Select training type
    trainSupervised = False
    trainPhysicsInformed = True

    # ray tune study 
    ray_tune_study = False

    # select what to visualize
    showFEMData = True
    showTrainingData = False


#%%  -------------------------------  Generating / collecting data ---------------------------------

    # get Paths
    pathDict = utils.getProjectPaths(os.path.dirname(os.path.realpath(__file__))[:-4])
    data = cdp.loadComsolDisplacements(pathDict, fileName)

    # Create Comsol data object and extract process raw data
    SimulationData = cdp.ComsolData(data)
    SimulationData.ExtractInformation()

    nodeKoord = SimulationData.nodeKoord
    nodeDisplacements = SimulationData.nodeDisplacements
    freqRange = SimulationData.frequencyRange

    X = nodeKoord
    U = nodeDisplacements[FREQUENCY] * scalingFEMData

    # Plot FEM data
    if showFEMData:
        vis_FEM = visgen.Visualizer(X,U, SimulationData.dimension)
        title = 'FEM nodes with deformation'
        _ , ax1 = vis_FEM.plot_with_deformation(AMPLFICATION, title)

        # plot the areas of the crossections used for visualization of results
        yy, zz = np.meshgrid(np.linspace(-0.2, 1.2, 12), np.linspace(-0.2, 1.2, 12)) 
        x = np.ones_like(yy)*0.22
        ax1.plot_surface(x, yy, zz, alpha=0.3)
        x = np.ones_like(yy)*0.51
        ax1.plot_surface(x, yy, zz, alpha=0.3)
        x = np.ones_like(yy)*0.68
        ax1.plot_surface(x, yy, zz, alpha=0.3)

        ax1.set_box_aspect([1,1,1])
        plt.show()


    # Get surface data used for training
    X_surf1_idx = np.where((X[:,0] == 0))[0]
    X_surf2_idx = np.where((X[:,0] == 0.8))[0]

    X_surf3_idx = np.where((X[:,1] == 0))[0]
    X_surf4_idx = np.where((X[:,1] == 1))[0]

    X_surf5_idx = []
    X_surf6_idx = []

    tol = 0.001
    for i,x in enumerate(X):
        x_curr = x[0]
        z_curr = x[2]
        
        # testing if point is on one of the surfaces
        z_upper = 0.33375 * x_curr + 0.733
        z_lower = 0.91625 * x_curr
        
        if abs(z_upper - z_curr) < tol:
            X_surf5_idx.append(i)
        elif abs(z_lower - z_curr) < tol:
            X_surf6_idx.append(i)
        
    X_surf5_idx = np.array(X_surf5_idx)
    X_surf6_idx = np.array(X_surf6_idx)

    X_surf_idx = np.concatenate((X_surf1_idx, X_surf2_idx, \
                                X_surf3_idx, X_surf4_idx, \
                                X_surf5_idx, X_surf6_idx, \
                                ), axis = 0)
        
    X_surf = X[X_surf_idx]
    U_surf = U[X_surf_idx]

    # plot the surface data 
    if showTrainingData:
        vis_surf = visgen.Visualizer(X_surf, U_surf, SimulationData.dimension)
        title = 'Surface nodes used for training'
        fig, ax = vis_surf.plot_with_deformation(0, title)
        ax.set_box_aspect([1,1,1])


    # Create collocation point sampler
    def CPSampler_3D(NUMBER_COLLOCATION_POINTS):
        # sample points in the domain and on the boundary in [m]
        x_range = [0, 0.8]
        y_range = [0, 1]
        z_range = [0, 1]

        # Collocation points points in the domain
        X_coll = utils.pointsInRange3D(round(NUMBER_COLLOCATION_POINTS * 0.66), x_range, y_range, z_range)
        
        keep_list = []
        # Eliminate all collocation points that are not in the domain 
        for i,x in enumerate(X_coll):
            x_curr = x[0]
            z_curr = x[2]
            
            # testing if point is on one of the surfaces
            
            z_upper = 0.33375 * x_curr + 0.733
            z_lower = 0.91625 * x_curr
            
            if (z_curr <= z_upper) and (z_curr >= z_lower):
                keep_list.append(i)

        X_coll = X_coll[keep_list]
        X_train_CP = torch.from_numpy(X_coll).float()
        
        return X_train_CP


    NUMBER_COLLOCATION_POINTS = 21000 # ca. 1/3 will stay 
    X_train_CP = CPSampler_3D(NUMBER_COLLOCATION_POINTS)


    # plot collocation points
    if showTrainingData:
        X_train_CP_np = X_train_CP.detach().numpy()
        visgen.scatter_3D_points(X_train_CP_np, 'Collocation points', color = 'tab:green', alpha = 0.2, colorbar = False)
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

    X_train = torch.from_numpy(X_surf).float()
    U_train = torch.from_numpy(U_surf).float() 

    X_test = torch.from_numpy(X).float()
    U_test = torch.from_numpy(U).float() 

    dt = Dataset()
    dt.X_train_CP = X_train_CP
    dt.X_train_BC_Dir = X_train
    dt.U_train_BC_Dir = U_train
    dt.X_test = X_test
    dt.U_test = U_test


#%% ------------------------------  Setting up the model and training ------------------------------

    inputSize = 3
    outputSize = 3

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
            pinn = NavierLame3DPINN(pdeParams, model, CPSampler_3D, None, w, True)
            pinn.train_model(hp, dt)

            # adding stuff to tensorboard
            hyperparameter_dict = hp.get_param_dict()
            metric_dict = pinn.get_final_metric_dict()
            w.writer.add_hparams(hyperparameter_dict, metric_dict, run_name = "res")
        
            # close writer        
            w.writer.close()
            
            
        path_to_studies = pathDict.get('pathToModelStudies') + "/3Dnavier_lame" 

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
        hp.optimizer = "Adam"
        hp.learningRate = 0.005
        hp.numEpochs = 5000
        hp.learningRateDecay = 0.5
        hp.numEpochsForDecay = 800
        hp.numNeuronsPerLayer = 20
        hp.numLayers = 12
        hp.batchSize = 500
        
        model = neu.NeuralNetwork(inputSize, outputSize, hp.numLayers, hp.numNeuronsPerLayer)
        training_loss, validation_loss = neu.trainNetworkSupervised(model, hp, X_train, U_train, X_test, U_test)
        
        
        loss_names = ['Training loss', 'Validation loss']
        training_losses = np.concatenate((np.array(training_loss).reshape(len(training_loss), 1),
                                        np.array(validation_loss).reshape(len(validation_loss), 1)), axis=1)
        fig1, ax1 = visnn.plotMultiLearningCurves(training_losses, loss_names, True)
        
                 
    elif (not ray_tune_study and trainPhysicsInformed):

        hp = neu.Hyperparameter()
        hp.optimizer = 'Adam'
        hp.learningRate = 0.001
        hp.numEpochs = 10
        hp.learningRateDecay = 0.5
        hp.numEpochsForDecay = 500
        hp.batchSize = 250
        hp.numNeuronsPerLayer = 12
        hp.numLayers = 20
        hp.scaling_strategy = 'Balance'

        # creating and training model
        model = neu.NeuralNetwork(inputSize, outputSize, hp.numLayers, hp.numNeuronsPerLayer)
        pinn = NavierLame3DPINN(pdeParams, model, CPSampler_3D)
        pinn.train_model(hp, dt)
        
        # extracting and plotting loss courves
        training_losses = np.concatenate((np.array(pinn.training_losses.training_loss_CP).reshape(len(pinn.training_losses.training_loss), 1),
                                        np.array(pinn.training_losses.training_loss_BC).reshape(len(pinn.training_losses.training_loss), 1)), axis=1)
        
        pde_loss = r'$\mathcal{L}_{\Omega,i} /\mathcal{L}_{\Omega,1} $'
        bc_loss_dir = r'$\mathcal{L}_{\Gamma_{u},i} /\mathcal{L}_{\Gamma_{u},1} $'
        
        loss_names = [pde_loss, bc_loss_dir]
        fig1, ax1 = visnn.plotMultiLearningCurves(training_losses, loss_names, True)
        
            
#%% --------------------------------------  Save and load model ------------------------------------
    if not ray_tune_study: 

        save_model  = False
        if save_model:
            folderName = "3Dnavier_lame"
            modelName = '3D_Ex2_ExcitHigh_10DataScaling_2180Hz_12Layers_20NeuPerLayer_SupervisedOnly_XXX'
            savePath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
            torch.save(model.state_dict(), savePath)
        
        
        '''
        High excitation:
        - The prescribed displacement (BC) is 10 times higher compared to Low excitation model 
        - Linear elastic case -> the whole resulting displacemet field mulitplied by factor 10 compared to Low excitation model 
        - High excitation PINN can be used for low excitation prediction -> scaling of prediction with 1/10 (data_scaling = 10)
        
        Low excitation:
        - Dirichlet BC (Displacements) are mulitplied with factor before training if using LBFGS (Necessary for LBFGS to converge)
        - PINN predictions of these models have to be devided by the same factor (data_scaling = factor)
        - Used factor is always included in model name / study_info.txt file
        - Low excitation PINN can be used for high excitation prediction -> scaling of prediction with 10 (data_scaling = 0.1)
        '''
        
        load_model = True
        if load_model:
            folderName = "3Dnavier_lame"
            
            # ONN
            modelName = '3D_Ex2_ExcitHigh_10DataScaling_2180Hz_12Layers_20NeuPerLayer_SupervisedOnly'
            scalingPredictions = 0.01 # for SupervisedOnly model

            # PINN
            modelName = "3D_Ex2_ExcitHigh_Adam_Batchmode_MeshFine_OnlyBoundary_2180Hz_12Layers_20NeuPerLayer_Study_7_Run_16"
            scalingPredictions = 0.1 # for PINN
            
            model = neu.NeuralNetwork(3, 3, 12, 20)
            loadPath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
            model.load_state_dict(torch.load(loadPath))
            
            

#%% ------------------------------ MAE on boundary and in the domain --------------------------------
                
        # nodes in the domain  ->  Little bit buggy, still some surface points within the domain points
        X_ges_idx_list = range(2310)
        X_surf_idx_list = list(X_surf_idx)
        X_domain_idx_list = list(set(X_ges_idx_list) - set(X_surf_idx_list))
        X_domain = X[X_domain_idx_list]
        U_domain = U[X_domain_idx_list] 
        X_boundary = X[X_surf_idx_list]
        U_boundary = U[X_surf_idx_list] 
        
        # Calculate the NN performance based on FEM points in the domain 
        MAE_func = torch.nn.L1Loss()
        MSE_func = torch.nn.MSELoss()
        
        U_pred_domain = model(torch.from_numpy(X_domain).float()) * scalingPredictions
        MAE = MAE_func(U_pred_domain, torch.from_numpy(U_domain).float()).detach().numpy()
        MSE = MSE_func(U_pred_domain, torch.from_numpy(U_domain).float()).detach().numpy()
        
        # Mean displacements in the dataset
        U_mean = (np.mean(abs(U_domain[:,0]))+np.mean(abs(U_domain[:,1]))+np.mean(abs(U_domain[:,2])))/3
        
        # Mean deviation of NN predictions in domain compared to FEM solution
        dev = MAE/U_mean * 100
        
        print('')
        print('------------- Performance for points in domain -----------------')
        print('Mean absolute error of predictions: ' + str(MAE) + '[m]')
        print('Mean squared error of predictions: ' + str(MSE) + '[m]')
        print('Mean deviation of predictions: ' + str(round(dev,2)) + '[%]')
        print('')
        
        
        # Calculate the NN performance based on FEM points on the boundary (Training data)
        U_pred_boundary = model(torch.from_numpy(X_boundary).float()) * scalingPredictions
        MAE = MAE_func(U_pred_boundary, torch.from_numpy(U_boundary).float()).detach().numpy()
        MSE = MSE_func(U_pred_boundary, torch.from_numpy(U_boundary).float()).detach().numpy()
        
        # Mean displacements in the dataset
        U_mean = (np.mean(abs(U_boundary[:,0]))+np.mean(abs(U_boundary[:,1]))+np.mean(abs(U_boundary[:,2])))/3
        
        # Mean deviation of NN predictions in domain compared to FEM solution
        dev = MAE/U_mean * 100
        
        print('')
        print('------ Performance for points on boundary (Training data) ------')
        print('Mean absolute error of predictions: ' + str(MAE) + '[m]')
        print('Mean squared error of predictions: ' + str(MSE) + '[m]')
        print('Mean deviation of predictions: ' + str(round(dev,2)) + '[%]')
        print('')


#%% ----------------------------  testing the network and plotting results -------------------------
        
        if True:
            #----------------------- Crossection in x - plane ----------------------------
            
            # 0.22, 0.51, 0.68
            const_koord = 0
            const_koord_min = 0.51 #0.22 #0.51 # 0.68 #
            const_koord_max = 0.52 #0.23 #0.52 # 0.69 # 
            
            X_cross_idx = np.where((X[:,const_koord] > const_koord_min) & (X[:,const_koord] < const_koord_max))
            X_cross_3D = X[X_cross_idx]
            
            X_cross_2D = np.array([X_cross_3D[:,1], X_cross_3D[:,2]]).T
            
            xlabel = r'$x$' +  ' (m)'
            ylabel = r'$z$' +  ' (m)'
            
        
        if False:
            #----------------------- Crossection in y - plane ----------------------------
        
            # 0.32, 0.5, 0.82
            const_koord = 1
            const_koord_min = 0.30 #0.30 #0.53 #0.84
            const_koord_max = 0.31 #0.31 #0.54 #0.85
        
            X_cross_idx = np.where((X[:,const_koord] > const_koord_min) & (X[:,const_koord] < const_koord_max))
            X_cross_3D = X[X_cross_idx]
            X_cross_2D = np.array([X_cross_3D[:,0], X_cross_3D[:,2]]).T
        
            xlabel = r'$x$' +  ' (m)'
            ylabel = r'$z$' +  ' (m)'
        
        # plot FEM solution
        U_cross = U[X_cross_idx]
        U_cross_displ = (U_cross[:,0]**2 + U_cross[:,1]**2 + U_cross[:,2]**2)**0.5        
        field_name = r'$ \parallel \mathbf{u} \parallel_{2}$' + "(m)"
        tit = 'True displacements'
        fig, ax = visgen.plot_interpolated_squared_field(X_cross_2D, U_cross_displ, title = tit, levels = 100, field_name = field_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # plot PINN prediction
        X_cross_3D_tensor = torch.from_numpy(X_cross_3D).float()
        U_cross_pred = model(X_cross_3D_tensor).detach().numpy() * scalingPredictions
        U_cross_pred_disp = (U_cross_pred[:,0]**2 + U_cross_pred[:,1]**2 + U_cross_pred[:,2]**2)**0.5
        field_name = r'$ \parallel \hat{\mathbf{u}} \parallel_{2}$'+ " (m)"
        tit = 'Predicted displacements'
        fig, ax = visgen.plot_interpolated_squared_field(X_cross_2D, U_cross_pred_disp, title = tit, levels = 100, field_name = field_name,
                                                        norm_cbar = True, X_norm = X_cross_2D, U_norm = U_cross_displ)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel) 

        # plot deviation between FEM and PINN solution
        U_cross_diff = abs(U_cross_displ - U_cross_pred_disp)
        cmap = cm.gist_heat
        field_name = r'$| \parallel\mathbf{u}\parallel_{2} - \parallel\hat{\mathbf{u}}\parallel_{2} $' + " (m)"
        tit = 'Differences of displacements'
        fig, ax = visgen.plot_interpolated_squared_field(X_cross_2D, U_cross_diff, title = tit, cmap = cmap, levels = 100, field_name = field_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel) 
        
        # Plots of 1D line of 2D Crossection in YZ plane
        TITLE_FONT_STYLE = 'Arial'
        csfont = {'fontname':TITLE_FONT_STYLE}

        stepSize = 0.01

        const_koord_min_1D = 0.73 #0.5 # 0.73 # 0.82/0.76
        const_koord_max_1D = 0.74 #0.51 # 0.74 # 0.83/0.77

        X_cross_1D_idx = np.where((X_cross_2D[:,1] > const_koord_min_1D) & (X_cross_2D[:,1] < const_koord_max_1D))[0]
        X_cross_1D = X_cross_2D[X_cross_1D_idx]

        u_cross_1D_disp = U_cross_displ[X_cross_1D_idx]
        u_cross_1D_pred_disp = U_cross_pred_disp[X_cross_1D_idx]

        y_old = X_cross_1D[:,0]
        y_new = np.arange(0,1,stepSize)

        f_pred = interpolate.interp1d(y_old,u_cross_1D_pred_disp, kind = 'quadratic')
        u_pred_interp = f_pred(y_new)

        f_true = interpolate.interp1d(y_old,u_cross_1D_disp, kind = 'quadratic')
        u_true_interp = f_true(y_new)

        fig, ax  = plt.subplots()
        #ax.plot(y_new,u_pred_interp_PINN,'--.', c = 'tab:blue')
        ax.plot(y_new,u_pred_interp,'--', c = 'tab:green')
        ax.plot(y_new,u_true_interp,'-', c = 'tab:red')
        ax.legend(['PINN solution','ONN solution','FEM solution'])
        ax.grid('minor')
        ax.set_xlabel(r'$y$' +  ' (m)')
        ax.set_ylabel(r'$\parallel\mathbf{u}\parallel_{2}$' + rf'$(x = {const_koord_min}, y, z = {const_koord_min_1D})$' + " (m)")
        ax.set_title('Comparison between PINN and FEM solution',**csfont)
        plt.show()

#%% -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()


# %%








