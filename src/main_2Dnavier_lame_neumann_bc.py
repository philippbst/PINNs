
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
    from matplotlib.pyplot import subplots
    import numpy as np
    import torch
    from ray import tune
    from scipy import interpolate

    from tools.pde_information import NavierLameParams
    import neural_network.network as neu
    from pinn.navier_lame_2D_pinn import NavierLame2DPINN
    import tools.comsol_data_processor as cdp
    import tools.utils as utils
    import visualization.visualization_nn as visnn
    import visualization.visualization_general as visgen
    from visualization.visualization_general import plot_interpolated_squared_field
    from pinn.utils_pinn import TensorboardWriterAssistant, RayTuneSetup, Dataset

    
#%% ----------------------------------------- Settings ----------------------------------------------
    # Define the used solution and the desired training method
    FREQUENCY = 1000 #2500 #4000
    AMPLFICATION = 10

    # Save data as tensorboard run
    save_tensorboard_run = False

    # ray tune study 
    rayTuneStudy = False

    # FEM solution file name for addtional solution and evaluation
    fileName = '2DNavierLame_freq_DisplacementMeshExtremeFine_LowFreq.txt'

    # show training data
    showTrainingData = True

    # scaling FEM solution before training and PINN predictions
    scalingFEMData = 1
    scalingPredictions = 1
#%%  -------------------------------  Generating / collecting data ---------------------------------

    # get Paths
    pathDict = utils.getProjectPaths(os.path.dirname(os.path.realpath(__file__))[:-4])

    # simulation domain restrictions
    x_min = -0.5
    x_max = 0.5
    y_min = -0.25
    y_max = 0.25
    step_size = 0.005

    # Boundary - Input
    y_l = np.arange(y_min,y_max,step_size).reshape(np.arange(y_min,y_max,step_size).shape[0],1)
    x_l = np.ones_like(y_l) * x_min
    X_l = np.concatenate((x_l, y_l), axis = 1)

    y_r = np.arange(y_min,y_max,step_size).reshape(np.arange(y_min,y_max,step_size).shape[0],1)
    x_r = np.ones_like(y_r) * x_max
    X_r = np.concatenate((x_r, y_r), axis = 1)

    x_o = np.arange(x_min,x_max,step_size).reshape(np.arange(x_min,x_max,step_size).shape[0],1)
    y_o = np.ones_like(x_o) * y_max
    X_o = np.concatenate((x_o, y_o), axis = 1)

    x_u = np.arange(x_min,x_max,step_size).reshape(np.arange(x_min,x_max,step_size).shape[0],1)
    y_u = np.ones_like(x_u) * y_min
    X_u = np.concatenate((x_u, y_u), axis = 1)

    X_train_BC_Dir_1 = X_l # Feste einspannung
    X_train_BC_Dir_2 = X_r # Verschiebung in y richtung, x richtung fix

    # Boundary - Outputs
    U_train_BC_Dir_1 = np.zeros_like(X_train_BC_Dir_1)
    U_train_BC_Dir_2 = np.ones_like(X_train_BC_Dir_2) * 0.01
    U_train_BC_Dir_2[:,0] = 0


    # Combine data 
    X_train_BC_Dir = np.concatenate((X_train_BC_Dir_1, X_train_BC_Dir_2), axis = 0)
    U_train_BC_Dir = np.concatenate((U_train_BC_Dir_1, U_train_BC_Dir_2), axis = 0)

    X_train_BC_Neu = np.concatenate((X_o, X_u), axis = 0)
    Y_train_BC_Neu = np.zeros_like(X_train_BC_Neu) 


    # load FEM solution and get additional solution points
    data = cdp.loadComsolDisplacements(pathDict, fileName)

    SimulationData = cdp.ComsolData(data)
    SimulationData.ExtractInformation()

    nodeKoord = SimulationData.nodeKoord
    nodeDisplacements = SimulationData.nodeDisplacements

    X = nodeKoord
    U = nodeDisplacements[FREQUENCY][:, :2] * scalingFEMData

    node_nums = [1600,1700,3240, 4000, 5100, 6000]
                
    X_measure = X[node_nums,:]
    U_measure = U[node_nums,:]

    for i in range(4):
        X_measure = np.concatenate((X_measure, X_measure), axis = 0) 
        U_measure = np.concatenate((U_measure, U_measure), axis = 0) 

    X_train_BC_Dir = np.concatenate((X_train_BC_Dir, X_measure), axis = 0)
    U_train_BC_Dir = np.concatenate((U_train_BC_Dir, U_measure), axis = 0)


    # Sample collocation points in the domain and on the BC and define material properties
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

    NUMBER_COLLOCATION_POINTS = 3000
    X_train_CP = CPSampler_2D(NUMBER_COLLOCATION_POINTS)


    # Plot training data
    if showTrainingData:
        vis_FEM = visgen.Visualizer(X,U, SimulationData.dimension)
        title = 'FEM nodes with deformation'
        _, ax = vis_FEM.plot_with_deformation(0, title)
        ax.scatter(X[node_nums,0],X[node_nums,1], c = 'black')

        X_coll = torch.clone(X_train_CP).detach().numpy()
        _, ax = plt.subplots()
        ax.scatter(X_train_BC_Dir[:,0], X_train_BC_Dir[:,1], c = 'tab:blue', alpha = 0.5);
        #p = ax.scatter(X_train_BC_Dir_2[:,0], X_train_BC_Dir_2[:,1], c = 'tab:gray', alpha = 0.5);
        ax.scatter(X_train_BC_Neu[:,0], X_train_BC_Neu[:,1], c = 'tab:red', alpha = 0.5);
        ax.scatter(X_coll[:,0], X_coll[:,1], color = 'tab:green', alpha = 0.2);
        ax.set_title('Data points used for training')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        plt.show()


#%% ------------------- Defining PDE parameter and create dataset for PINN --------------------------
    E = 205e9                   # in [Pa = N/m^2]
    nu = 0.28

    lam = (nu/(1-2*nu)) * (1/(1+nu)) * E
    my = 1/2 * (1/(1+nu)) * E

    LAMBDA = lam               # in [N/m^2]
    MY = my                     # in [N/m^2]
    OMEGA = 2*np.pi*FREQUENCY   # 2*pi*FREQUENCY
    RHO = 7850                  # in [kg / m^3]

    pdeParams = NavierLameParams(LAMBDA, MY, RHO, OMEGA)

    # transform data to tensors and create dataset
    X_train_BC_Dir = torch.from_numpy(X_train_BC_Dir).float()
    U_train_BC_Dir = torch.from_numpy(U_train_BC_Dir).float()

    X_train_BC_Neu = torch.from_numpy(X_train_BC_Neu).float()
    Y_train_BC_Neu = torch.from_numpy(Y_train_BC_Neu).float()

    X_test = torch.from_numpy(X).float()
    U_test = torch.from_numpy(U).float()

    dt = Dataset()
    dt.X_train_CP = X_train_CP
    dt.X_train_BC_Neu = X_train_BC_Neu
    dt.Y_train_BC_Neu = Y_train_BC_Neu
    dt.X_train_BC_Dir = X_train_BC_Dir
    dt.U_train_BC_Dir = U_train_BC_Dir
    dt.X_test = X_test
    dt.U_test = U_test


#%% ------------------------------  Setting up the model and training ------------------------------
    inputSize = 2
    outputSize = 2

    if rayTuneStudy:
        
        # Testing config
        config = {
            "learningRate": tune.grid_search([0.1, 0.0001]),
            "numEpochs": tune.grid_search([5, 10]),
            "optimizer": tune.choice(["Adam"]),
            "learningRateDecay": tune.grid_search([0.3]),
            "numEpochsForDecay": tune.grid_search([500]),
            "batchSize": tune.grid_search([None]), # Corresponds to number of Dirichlet BC samples per training
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
            
            

        path_to_studies = pathDict.get('pathToModelStudies') + "/2Dnavier_lame_neumann" 

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
    
        
    else:
        hp = neu.Hyperparameter()
        hp.optimizer = 'Adam'
        hp.learningRate = 0.01 
        hp.numEpochs = 10000
        hp.learningRateDecay = 0.5 
        hp.numEpochsForDecay = 500
        hp.batchSize = 25 
        hp.numNeuronsPerLayer = 25
        hp.numLayers = 8
        hp.scaling_strategy = 'Manual'
        hp.weight_pde_loss.append(1e-21)
        hp.weight_dirichlet_loss.append(10000)
        hp.weight_neumann_loss.append(2e-24)

        # creating and training model
        model = neu.NeuralNetwork(inputSize, outputSize, hp.numLayers, hp.numNeuronsPerLayer)
        pinn = NavierLame2DPINN(pdeParams, model, CPSampler_2D, None)
        pinn.train_model(hp, dt)
        
        # extracting and plotting loss courves
        training_losses = np.concatenate((np.array(pinn.training_losses.training_loss_CP).reshape(len(pinn.training_losses.training_loss), 1),
                                        np.array(pinn.training_losses.training_loss_BC_Dir).reshape(len(pinn.training_losses.training_loss), 1),
                                        np.array(pinn.training_losses.training_loss_BC_Neu).reshape(len(pinn.training_losses.training_loss), 1)), axis=1)
    
        
        pde_loss = r'$\mathcal{L}_{\Omega,i} /\mathcal{L}_{\Omega,1} $'
        bc_loss_dir = r'$\mathcal{L}_{\Gamma_{u},i} /\mathcal{L}_{\Gamma_{u},1} $'
        bc_loss_neu = r'$\mathcal{L}_{\Gamma_{\sigma},i} /\mathcal{L}_{\Gamma_{\sigma},1} $'
        
        loss_names = [pde_loss, bc_loss_dir, bc_loss_neu]
        fig1, ax1 = visnn.plotMultiLearningCurves(training_losses, loss_names, True)
    


#%% --------------------------------------  Save and load model ------------------------------------
        save_model  = False
        if save_model:
            folderName = "2Dnavier_lame_neumann"
            modelName = '2D_navier_lame_freq_Neumann_BC_1000Hz_5SupportSolutionPoints'
            savePath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
            torch.save(model.state_dict(), savePath)


        load_model = True
        if load_model:
            folderName = "2Dnavier_lame_neumann"
            
            # First trained model without optimization for High Excitation 
            modelName = '2D_navier_lame_freq_Neumann_BC_1000Hz_5SupportSolutionPoints_Study1_Run14_8Layers_25NeuronsPerLayer'
            loadPath = pathDict['pathToTrainedModels'] + '/' + folderName + '/' + modelName + '.pt'
            
            model = neu.NeuralNetwork(2, 2, 8, 25)
            model.load_state_dict(torch.load(loadPath))
            
            pinn = NavierLame2DPINN(pdeParams, model, None, None)
                            
            # Scaling to predict LowExcit with HighExcit model
            scalingPredictions = 1

    
#%% ----------------------------  testing the network and plotting results -------------------------

        # Ignore the third dimension with z = 0 and scale solution
        U_ges = U [:,:2] 
        
        # make predictions with trained model
        U_pred = model(torch.from_numpy(X).float()).detach().numpy() * scalingPredictions
        
        # plot FEM solution 
        field_name = r'$ \parallel\mathbf{u}\parallel_{2}$' + r'$[m]$'
        U_ges = ((U[:,0])**2 + (U[:,1])**2)**0.5
        tit = 'FEM solution'
        fig, ax = plot_interpolated_squared_field(X, U_ges, levels = 100, field_name = field_name)
        ax.set_xlabel(r'$x [m]$')
        ax.set_ylabel(r'$y [m]$')
        csfont = {'fontname':'Arial'}
        plt.title(tit,**csfont)
        
        # plot PINN prediction
        field_name = r'$ \parallel\hat{\mathbf{u}}\parallel_{2}$' + r'$[m]$'
        U_pred_ges = ((U_pred[:,0])**2 + (U_pred[:,1])**2)**0.5
        tit = 'Solution predicted by the PINN'
        fig, ax = plot_interpolated_squared_field(X, U_pred_ges, levels = 100, field_name = field_name, \
                                                norm_cbar = True, X_norm = X, U_norm = U_ges)
        ax.set_xlabel(r'$x [m]$')
        ax.set_ylabel(r'$y [m]$')
        csfont = {'fontname':'Arial'}
        plt.title(tit,**csfont)
        
        # plot deviation between FEM and PINN solution
        field_name = r'$| \parallel\mathbf{u}\parallel_{2} - \parallel\hat{\mathbf{u}}\parallel_{2} |$' + r'$[m]$'
        dev = abs(U_pred_ges - U_ges)
        tit = 'Deviation'
        fig, ax = plot_interpolated_squared_field(X, dev, levels = 100, cmap = cm.gist_heat, field_name = field_name)
        ax.set_xlabel(r'$x [m]$')
        ax.set_ylabel(r'$y [m]$')
        csfont = {'fontname':'Arial'}
        plt.title(tit,**csfont)
        
        # Calculate the stresses and plot them 
        sigma_xx, sigma_xy, sigma_yy = pinn.calculate_cauchy_stresses(torch.from_numpy(X).float())
        mieses_spannungen = (sigma_xx**2 + sigma_yy**2 - sigma_xx*sigma_yy + 3*(sigma_xy**2))*0.5
        
        stresses_to_plot = sigma_yy.detach().numpy().reshape(sigma_xx.shape[0],)
        field_name = r'$ \sigma_{yy}  [N/m^{2}]$'
        tit = 'Stresses predicted by the PINN'
        fig, ax = plot_interpolated_squared_field(X, stresses_to_plot, levels = 200, field_name = field_name)
        ax.set_xlabel(r'$x [m]$')
        ax.set_ylabel(r'$y [m]$')
        csfont = {'fontname':'Arial'}
        plt.title(tit,**csfont)
        
        # Calculate the strains and plot them 
        eps_xx, eps_xy, eps_yy = pinn.calculate_linear_strains(torch.from_numpy(X).float()) 
        
        strains_to_plot = eps_yy.detach().numpy().reshape(eps_xx.shape[0],)
        field_name = r'$ \epsilon_{yy}$'
        tit = 'Strains predicted by the PINN'
        fig, ax = plot_interpolated_squared_field(X, strains_to_plot, levels = 200, field_name = field_name)
        ax.set_xlabel(r'$x [m]$')
        ax.set_ylabel(r'$y [m]$')
        csfont = {'fontname':'Arial'}
        plt.title(tit,**csfont)
        
        # Plots of 1D line 
        stepSize = 0.01
        const_koord_min_1D = -0.001
        const_koord_max_1D = 0.001
        
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
        ax.legend(['PINN solution','FEM solution'])
        ax.grid('minor')
        ax.set_xlabel(r'$x [m]$')
        ax.set_ylabel(rf'$u(x, y = 0) [m]$')
        ax.set_title('Comparison between PINN and FEM solution',**csfont)
        
        plt.show()

    
#%% ------------------------------ MAE on boundary and in the domain --------------------------------

        # MAE for domain
        idx_dom = np.where((abs(X[:, 0]) != 0.5) & (abs(X[:, 1]) != 0.25))[0]
        X_dom = X[idx_dom, :]
        U_dom = U[idx_dom, :]

        # Calculate measures
        MAE_func = torch.nn.L1Loss()
        MSE_func = torch.nn.MSELoss()
        
        U_pred = model(torch.from_numpy(X_dom).float()) * scalingPredictions 
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