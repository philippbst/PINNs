#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:02:36 2021

@author: philippbst
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from neural_network.loss_functions import MSEloss
from pinn.utils_pinn import record_time, Loss_curves
import pinn.utils_pinn as utils_pinn
from ray import tune


#Abtract base class to define the needed methods for PINN implementations
class GeneralPINN(ABC):
    def __init__(self, pdeParam, model, CPSampler=None, progressTracker=None, tensorboard_writer = None, ray_tune_study = False):
        self.pdeParam = pdeParam
        self.model = model
        self.collocationPointSampler = CPSampler
        self.progressTracker = progressTracker
        self.tensorboard_writer = tensorboard_writer
        self.ray_tune_study = ray_tune_study
        self.training_losses = Loss_curves([], [], [], [], [], [], [])

    @abstractmethod
    def evaluateResiduals(self, X):
        pass

    @abstractmethod
    def _calculateResiduals(self, u_pred_dict, u_deriv_dict):
        pass

    @abstractmethod
    def _forwardWithGrads(self, X):
        pass

    @abstractmethod
    def _physicsInformedLoss(self, X):
        pass
    
    @abstractmethod
    def _neumannBCLoss(self, X, Y, N = None):
        pass
    
    @abstractmethod
    def _robinBCLoss(self, X, Y, N = None):
        pass
    

    def testModel(self):
        ''' 
        -> TO DO
        '''
        pass
    
    
    def get_final_metric_dict(self):
        if (len(self.training_losses.training_loss) != 0):
            metric_dict = {'final/loss_train_ges': self.training_losses.training_loss[-1],\
                           'final/loss_train_BC': self.training_losses.training_loss_BC[-1],\
                           'final/loss_train_CP': self.training_losses.training_loss_CP[-1]}  
            if (len(self.training_losses.testing_loss) != 0):
                metric_dict['final/loss_test'] = self.training_losses.testing_loss[-1]
   
        return metric_dict
        
    
    ''' Following scaling strategies are implemented
        scaling = 'Annealing', 'SoftAdapt', 'Manual', 'Balance'
    '''
            
    @record_time
    def train_model(self, hyperparameter, dataset):
        
        if hyperparameter.scaling_strategy not in ['Annealing', 'SoftAdapt', 'Manual', 'Balance']:
            raise ValueError('Invalid scaling strategy chosen, select between: Annealing, SoftAdapt, Manual, Balance')
        
        if (hyperparameter.optimizer == "Adam"):
            if hyperparameter.batchSize is None:
                print("\n >>> Training model with Adam \n")
                self.__train_model_Adam(hyperparameter, dataset)
                
            else:
                print("\n >>> Training model with Adam with mini batches \n")
                self.__train_model_Adam_Batches(hyperparameter, dataset)

                    
        elif (hyperparameter.optimizer == "LBFGS"):
            print("\n >>> Training model with LBFGS \n")
            self.__train_model_LBFGS(hyperparameter, dataset)

            
        else:
            raise ValueError("Defined optimizer not yet implemented, so far only Adam and LBFGS available")
        
        return None


    # private methods
    def __train_model_Adam(self, hyperparameter, dataset):
        if not(dataset.X_train_BC_Neu is None):
            dataset.X_train_BC_Neu.requires_grad = True
            
        # tracking all operations on X_train_CP so that we can backpropagate through the network
        dataset.X_train_CP.requires_grad = True
        numCP = dataset.X_train_CP.shape[0]

        optimizer = Adam(self.model.parameters(), lr=hyperparameter.learningRate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        scheduler = StepLR(optimizer, step_size=hyperparameter.numEpochsForDecay,gamma=hyperparameter.learningRateDecay)

        if hyperparameter.scaling_strategy == 'Balance':
            self.__balance_scaling_factor(dataset, hyperparameter)

        for epoch in range(1, hyperparameter.numEpochs + 1):

            if not(self.collocationPointSampler is None):
                X_train_CP = self.collocationPointSampler(numCP)
                X_train_CP.requires_grad = True

            # update loss with scaling strategy
            if (epoch == 1) or (epoch%10 == 0):
                if hyperparameter.scaling_strategy == 'Annealing':
                    self.__update_lr_with_annealing(dataset, hyperparameter)
                elif hyperparameter.scaling_strategy == 'SoftAdapt':
                    self.__update_lr_with_softAdapt(hyperparameter)
                    
            # Setting gradients to zero
            optimizer.zero_grad()
                
            # executing one training itteration 
            ls = self.__forward_pass(dataset, hyperparameter)
                    
            # propagating backward
            ls.loss_train.backward()
            
            # updating parameters
            optimizer.step()

            # update learning rate each epoch
            scheduler.step()

            # save data and print progress to console
            self.__storing_and_showing_epoch_data(ls, epoch, dataset, hyperparameter)
        
        return None              
                                     
    def __train_model_Adam_Batches(self, hyperparameter, dataset):
        # tracking all operations on X_train_CP so that we can backpropagate through the network
        if not(dataset.X_train_BC_Neu is None):
            dataset.X_train_BC_Neu.requires_grad = True
        
        dataset.X_train_BC_Dir.requires_grad = True
        
        dataset.X_train_CP.requires_grad = True
        numCP = dataset.X_train_CP.shape[0]

        optimizer = Adam(self.model.parameters(), lr=hyperparameter.learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        scheduler = StepLR(optimizer, step_size=hyperparameter.numEpochsForDecay,gamma=hyperparameter.learningRateDecay)

        if hyperparameter.scaling_strategy == 'Balance':
            self.__balance_scaling_factor(dataset, hyperparameter)
        
        # creating data loader for batch training
        xDim = dataset.X_train_BC_Dir.shape[1]
        dataSet_ICBC = torch.cat([dataset.X_train_BC_Dir, dataset.U_train_BC_Dir], axis=1)
        data_BC_Dir = DataLoader(dataSet_ICBC, batch_size=hyperparameter.batchSize, shuffle=True)
        numBatches = np.ceil(dataset.X_train_BC_Dir.shape[0] / hyperparameter.batchSize)
        batchSize_CP = int(np.ceil(dataset.X_train_CP.shape[0] / numBatches))
        data_CP = DataLoader(dataset.X_train_CP, batch_size=batchSize_CP)
        
        if not(dataset.X_train_BC_Neu is None):
            yDim = dataset.X_train_BC_Neu.shape[1]
            dataSet_BC_Neu = torch.cat([dataset.X_train_BC_Neu, dataset.Y_train_BC_Neu], axis=1)
            batchSize_BC_Neu = int(np.ceil(dataset.X_train_BC_Neu.shape[0]/numBatches))
            data_BC_Neu = DataLoader(dataSet_BC_Neu, batch_size=batchSize_BC_Neu, shuffle=True)
        else:
            data_BC_Neu = data_BC_Dir
            

        for epoch in range(1, hyperparameter.numEpochs + 1):

            if not(self.collocationPointSampler is None):
                # sampling CP in every epoch if a sampler exists
                X_train_CP = self.collocationPointSampler(numCP)
                X_train_CP.requires_grad = True
                batchSize_CP = int(np.ceil(X_train_CP.shape[0]/numBatches))
                data_CP = DataLoader(X_train_CP, batch_size=batchSize_CP)

        
            for batch_number, (d_BC_Dir, d_CP, d_BC_Neu) in enumerate(zip(data_BC_Dir, data_CP, data_BC_Neu)):
                X_train_BC_Dir = d_BC_Dir[:, 0:xDim]
                U_train_BC_Dir = d_BC_Dir[:, xDim:]
                X_train_CP = d_CP
                
                if not(dataset.X_train_BC_Neu is None):
                    X_train_BC_Neu = d_BC_Neu[:, 0:xDim]
                    Y_train_BC_Neu = d_BC_Neu[:, xDim:]

                # update loss with scaling strategy
                if (batch_number == 0) and ((epoch == 1) or (epoch%10 == 0)):
                    if hyperparameter.scaling_strategy == 'Annealing':
                        self.__update_lr_with_annealing(dataset, hyperparameter)
                    elif hyperparameter.scaling_strategy == 'SoftAdapt':
                        self.__update_lr_with_softAdapt(hyperparameter)
                
                # Setting gradients to zero
                optimizer.zero_grad()
               
                # executing one training itteration 
                ls = self.__forward_pass(dataset, hyperparameter)


                '''
                #Can be used to visulaize the gradients of each layer to check vanashing gradients of different loss terms
                
                if (epoch ==1 or (epoch % 6000 == 0)) and (batch_number == 1):
                    import matplotlib.pyplot as plt
                    import scipy.stats as stats
                    
                    for j in range(hyperparameter.numLayers): # da hp.numLayers = 5
                    
                        # propagating backward
                        loss_CP.backward(retain_graph=True)
                        grad_CP = torch.clone(self.model.layers[j].weight.grad)
                        grad_CP = grad_CP.reshape(grad_CP.shape[0] * grad_CP.shape[1],1)
                        self.model.layers[j].zero_grad()
                        
                        # propagating backward
                        loss_BC_Dir.backward(retain_graph=True)
                        grad_BC_Dir = torch.clone(self.model.layers[j].weight.grad)
                        grad_BC_Dir = grad_BC_Dir.reshape(grad_BC_Dir.shape[0] * grad_BC_Dir.shape[1],1)
                        self.model.layers[j].zero_grad()
                        
                        # propagating backward
                        loss_BC_Neu.backward(retain_graph=True)
                        grad_BC_Neu = torch.clone(self.model.layers[j].weight.grad)
                        grad_BC_Neu = grad_BC_Neu.reshape(grad_BC_Neu.shape[0] * grad_BC_Neu.shape[1],1)
                        self.model.layers[j].zero_grad()
                        
                        ax, fig = plt.subplots()
                        n1, x1, _ = plt.hist(grad_CP.T,50,histtype=u'step', density=True)
                        n2, x2, _ = plt.hist(grad_BC_Dir.T,50,histtype=u'step', density=True)
                        n3, x3, _ = plt.hist(grad_BC_Neu.T,50,histtype=u'step', density=True)
                        plt.legend(['dL_CP/dw', 'dL_BC_Dir/dw', 'dL_BC_Neu/dw'])
                        plt.title(f'Layer: {j}')
                        plt.autoscale()  
                        plt.grid('minor')
                        plt.show()
                        
                    if hyperparameter.numLayers == 5:
                        dy = 400
                        dx = 700
                        
                        plt.figure(2)
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(0, 0, dx, dy)
                        plt.show
                        
                        plt.figure(3)
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(dx, 0, dx, dy)
                        plt.show
                        
                        plt.figure(4)
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(0, dy+50, dx, dy)
                        plt.show
                        
                        plt.figure(5)
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(dx, dy+50, dx, dy)
                        plt.show
                        
                        plt.figure(1)
                        mngr = plt.get_current_fig_manager()
                        mngr.window.setGeometry(dx/2 + 50, dy+50, dx-100, dy-100)
                        plt.show
    
                    stopper = 1
                    #plt.close('all')
                '''
                
                # propagating backward
                ls.loss_train.backward()
                
                # updating parameters
                optimizer.step()

            # update learning rate each epoch
            scheduler.step()

            # save data and print progress to console
            self.__storing_and_showing_epoch_data(ls, epoch, dataset, hyperparameter)
                       
        return None

    def __train_model_LBFGS(self, hyperparameter, dataset):
        # tracking all operations on X_train_CP so that we can backpropagate through the network
        if not(dataset.X_train_BC_Neu is None):
            dataset.X_train_BC_Neu.requires_grad = True
        
        dataset.X_train_CP.requires_grad = True
        numCP = dataset.X_train_CP.shape[0]

        optimizer = LBFGS(self.model.parameters(),lr=hyperparameter.learningRate)

        if hyperparameter.scaling_strategy == 'Balance':
            self.__balance_scaling_factor(dataset, hyperparameter)
        
        for epoch in range(1, hyperparameter.numEpochs + 1):
            if not(self.collocationPointSampler is None):
                X_train_CP = self.collocationPointSampler(numCP)
                X_train_CP.requires_grad = True


            # update loss with scaling strategy
            if (epoch == 1) or (epoch%10 == 0):
                if hyperparameter.scaling_strategy == 'Annealing':
                    self.__update_lr_with_annealing(dataset, hyperparameter)
                elif hyperparameter.scaling_strategy == 'SoftAdapt':
                    self.__update_lr_with_softAdapt(hyperparameter)
            
            
            def closure():
                # Setting gradients to zero
                optimizer.zero_grad()

                # executing one forward pass with loss calculation
                ls = self.__forward_pass(dataset, hyperparameter)
            
                # propagating backward
                ls.loss_train.backward()
                
                return ls.loss_train

            # updating parameters
            optimizer.step(closure)
            
            # repeat model prediction to track loss behaviour
            ls = self.__forward_pass(dataset, hyperparameter)

            
            # stop training if it diverges
            increase_limit = 100
            if epoch > 1:
                if ls.loss_train > (self.training_losses.training_loss[-1] * increase_limit) or np.isnan(ls.loss_train.detach().numpy()):
                    print("\n --------------------- ATTENTION ----------------------")
                    print(f"\n increase of loss by more than factor {increase_limit} in epoch {epoch}")
                    print("\n training diverged and not completed succesfully \n")
                    break
            
            # save data and print progress to console
            self.__storing_and_showing_epoch_data(ls, epoch, dataset, hyperparameter)
                   
        return None
   
    # private utils methods called in training functions above
    def __balance_scaling_factor(self, dataset, hyperparameter):

        # computing the physics informed loss of the predictions at the collocation points
        loss_CP = self._physicsInformedLoss(dataset.X_train_CP).detach().numpy()

        # predicting the model output for given data points (IC_BC)
        U_pred_ICBC = self.model(dataset.X_train_BC_Dir)

        # computing MSE loss for points with given solution
        loss_BC_Dir = MSEloss(dataset.U_train_BC_Dir, U_pred_ICBC).detach().numpy()

        ''' SCALING has to be adapted properly  '''
        scaling_pde = (loss_BC_Dir / loss_CP) * 0.1 

        if dataset.X_train_BC_Neu != None:
            loss_BC_Neu = self._neumannBCLoss(dataset.X_train_BC_Neu, dataset.Y_train_BC_Neu).detach().numpy()
            # Scaling Neumann BC loss to be equal 0 times higher than Dirichlet BC loss in first epoch
            ''' SCALING_BC_NEU has to be adapted properly  '''
            scaling_neumann_bc = (loss_BC_Dir / loss_BC_Neu) * 1 
            hyperparameter.weight_neumann_loss.append(scaling_neumann_bc)
                
        hyperparameter.weight_dirichlet_loss.append(1)
        hyperparameter.weight_pde_loss.append(scaling_pde)
        
        return None 
    
    
    def __update_lr_with_softAdapt(self, hyperparameter):
        '''
        
        NOT YET CORRECTLY IMPLEMENTED
        
        '''
        raise('Not yet correct implemented')
        
        if len(self.training_losses.training_loss_CP) >= 2:
                
            pde = self.training_losses.training_loss_CP[len(self.training_losses.training_loss_CP)-1] / self.training_losses.training_loss_CP[len(self.training_losses.training_loss_CP)-2]
            diri = self.training_losses.training_loss_BC_Dir[len(self.training_losses.training_loss_BC_Dir)-1] / self.training_losses.training_loss_BC_Dir[len(self.training_losses.training_loss_BC_Dir)-2]
            neum = self.training_losses.training_loss_BC_Neu[len(self.training_losses.training_loss_BC_Neu)-1] / self.training_losses.training_loss_BC_Neu[len(self.training_losses.training_loss_BC_Neu)-2]

            hyperparameter.weight_pde_loss.append(np.exp(pde) / (np.exp(pde) +  np.exp(diri) + np.exp(neum)) *  hyperparameter.weight_pde_loss)
            hyperparameter.weight_dirichlet_loss.append(np.exp(diri) / (np.exp(pde) +  np.exp(diri) + np.exp(neum)) * hyperparameter.weight_dirichlet_loss)
            hyperparameter.weight_neumann_loss.append(np.exp(neum) / (np.exp(pde) +  np.exp(diri) + np.exp(neum)) * hyperparameter.weight_neumann_loss)

        
        else:
            # No update possible before second epoch
            pass
        
        return None
        
    def __update_lr_with_annealing(self, dataset, hyperparameter):

        # computing the physics informed loss of the predictions at the collocation points
        loss_CP = self._physicsInformedLoss(dataset.X_train_CP)

        end = len(hyperparameter.weight_pde_loss) - 1 
        
        # scaling the PDE to make sure it is not too big
        loss_CP = loss_CP * hyperparameter.weight_pde_loss[end]
        
        # predicting the model output for given data points (IC_BC)
        U_pred_ICBC = self.model(dataset.X_train_BC_Dir)

        # computing MSE loss for points with given solution
        loss_BC_Dir = MSEloss(dataset.U_train_BC_Dir, U_pred_ICBC) * hyperparameter.weight_dirichlet_loss[end]

        # computing Neumann BC loss
        loss_BC_Neu = self._neumannBCLoss(dataset.X_train_BC_Neu, dataset.Y_train_BC_Neu) * hyperparameter.weight_neumann_loss[end]
        
        # update learning rate
        self.__learning_rate_annealing_algorithm(loss_CP, loss_BC_Dir, loss_BC_Neu, hyperparameter)
            
        
        return None
    
    def __learning_rate_annealing_algorithm(self, loss_CP, loss_BC_Dir, loss_BC_Neu, hyperparameter):
        '''
        from: https://arxiv.org/abs/2001.04536
        '''
        
        alpha = 0.9
        
        
        # Hyperparameter introduced by me to avoid exploding correcting factors
        #gamma = 0.01
        gamma = 1
        
        ''' Original version of annealing with added gamma correction factor '''
        # propagating PDE loss backward
        loss_CP.backward(retain_graph=True)
        
        # get max absolute value of gradients
        max_abs_grad_PDE = 0
        for j in range(hyperparameter.numLayers): 
            grad_CP = torch.abs(torch.clone(self.model.layers[j].weight.grad))
            if torch.max(grad_CP) > max_abs_grad_PDE:
                max_abs_grad_PDE = torch.max(grad_CP)
            self.model.layers[j].zero_grad()
        

        # # ''' Try out trainig with this adapted annealing version '''
        # # adapt learning rate annealing - not use max grad of PDE loss but also mean loss like for BCs
        # loss_CP.backward(retain_graph=True)
        # sum_abs_mean_layers_PDE = 0
        # for j in range(hyperparameter.numLayers): 
        #     grad_PDE = torch.abs(torch.clone(self.model.layers[j].weight.grad))
        #     grad_PDE_mean = torch.mean(grad_PDE)
        #     sum_abs_mean_layers_PDE = sum_abs_mean_layers_PDE + grad_PDE_mean
        #     self.model.layers[j].zero_grad()
        # mean_abs_grad_PDE = sum_abs_mean_layers_PDE / hyperparameter.numLayers  
        # max_abs_grad_PDE = mean_abs_grad_PDE
        
        
        ''' here original algorithm goes on '''
        # propagating Dirichlet BC loss backward
        loss_BC_Dir.backward(retain_graph=True)
        sum_abs_mean_layers_dir = 0
        for j in range(hyperparameter.numLayers): 
            grad_BC_Dir = torch.abs(torch.clone(self.model.layers[j].weight.grad))
            grad_BC_Dir_mean = torch.mean(grad_BC_Dir)
            sum_abs_mean_layers_dir = sum_abs_mean_layers_dir + grad_BC_Dir_mean
            self.model.layers[j].zero_grad()
        mean_abs_grad_Dir_BC = sum_abs_mean_layers_dir / hyperparameter.numLayers  
        
        # propagating Neumann BC loss backward
        loss_BC_Neu.backward(retain_graph=True)
        sum_abs_mean_layers_neu = 0
        for j in range(hyperparameter.numLayers): 
            grad_BC_Neu = torch.abs(torch.clone(self.model.layers[j].weight.grad))
            grad_BC_Neu_mean = torch.mean(grad_BC_Neu)
            sum_abs_mean_layers_neu = sum_abs_mean_layers_neu + grad_BC_Neu_mean
            self.model.layers[j].zero_grad()
        mean_abs_grad_Neu_BC = sum_abs_mean_layers_neu / hyperparameter.numLayers  
        
        scaling_bc_dir_hat = (max_abs_grad_PDE / mean_abs_grad_Dir_BC) * gamma 
        scaling_bc_neu_hat = (max_abs_grad_PDE / mean_abs_grad_Neu_BC) * gamma
        

        end = len(hyperparameter.weight_pde_loss) - 1 
        hyperparameter.weight_dirichlet_loss.append((1-alpha) * hyperparameter.weight_dirichlet_loss[end] + alpha * scaling_bc_dir_hat * hyperparameter.weight_dirichlet_loss[end]) 
        hyperparameter.weight_neumann_loss.append((1-alpha) * hyperparameter.weight_neumann_loss[end] + alpha * scaling_bc_neu_hat * hyperparameter.weight_neumann_loss[end])
        hyperparameter.weight_pde_loss.append(hyperparameter.weight_pde_loss[end])
        return None
        
         
        
    def __forward_pass(self, dataset, hyperparameter):
        
        ls = utils_pinn.Losses()
        end = len(hyperparameter.weight_pde_loss) - 1 
        
        # computing the physics-informed loss (PDE loss) of the predictions at the collocation points
        loss_CP = self._physicsInformedLoss(dataset.X_train_CP) * hyperparameter.weight_pde_loss[end]
        
        # predicting the model output for given data points (Dirichlet BC and points with prescribed solution)
        U_pred_BC_dir = self.model(dataset.X_train_BC_Dir) 

        # computing Dirichlet BC loss based on points with given solution
        loss_BC_Dir = MSEloss(dataset.U_train_BC_Dir, U_pred_BC_dir) * hyperparameter.weight_dirichlet_loss[end]

        # calculating Neumann Loss if Neumann BC training data available
        if dataset.X_train_BC_Neu != None:
            loss_BC_Neu = self._neumannBCLoss(dataset.X_train_BC_Neu, dataset.Y_train_BC_Neu) * hyperparameter.weight_neumann_loss[end]
            ls.loss_BC_Neu = loss_BC_Neu
            
        else:
            loss_BC_Neu = 0
      
        # calculating mixed BC loss if mixed BC training data are available
        if dataset.X_train_BC_Robin != None:
            loss_BC_Robin = self._robinBCLoss(dataset.X_train_BC_Robin, dataset.Y_train_BC_Robin) * hyperparameter.weight_robin_loss[end]
            ls.loss_BC_Robin = loss_BC_Robin
            
        else:
            loss_BC_Robin = 0
        
        # adding loss terms together and storing in dataclass
        ls.loss_train = loss_CP + loss_BC_Dir + loss_BC_Neu + loss_BC_Robin
        ls.loss_BC = loss_BC_Dir + loss_BC_Neu + loss_BC_Robin
        ls.loss_CP = loss_CP
        ls.loss_BC_Dir = loss_BC_Dir
        
        return ls
    
    
    def __storing_and_showing_epoch_data(self, ls, epoch, dataset, hyperparameter):
        # save data
        self.training_losses.training_loss_CP.append(ls.loss_CP.detach().numpy())
        self.training_losses.training_loss_BC.append(ls.loss_BC.detach().numpy())
        self.training_losses.training_loss.append(ls.loss_train.detach().numpy())

        testing = (dataset.X_test != None)
        
        if dataset.X_test != None:
            # calculate validation loss
            U_pred_test = self.model(dataset.X_test)
            loss_test = MSEloss(dataset.U_test, U_pred_test)
            self.training_losses.testing_loss.append(loss_test.detach().numpy())
            
        if dataset.X_train_BC_Neu!= None:
            self.training_losses.training_loss_BC_Dir.append(ls.loss_BC_Dir.detach().numpy())
            self.training_losses.training_loss_BC_Neu.append(ls.loss_BC_Neu.detach().numpy())


        if ((epoch == 1) | (epoch % 50 == 0)):
            self.__print_training_progress(testing, epoch)
            
        # add data to progress tracker
        if (self.progressTracker != None) and ((epoch == 1) | (epoch % self.progressTracker.epochStep == 0)):
            self.progressTracker.track(epoch, self.model)
                    
        # add data to tensorboard run
        if (self.tensorboard_writer != None):
            if dataset.X_test != None:
                self.tensorboard_writer.add_content(\
                        epoch, testing, self.training_losses.training_loss[-1],\
                        self.training_losses.training_loss_BC[-1], self.training_losses.training_loss_CP[-1],\
                        self.training_losses.testing_loss[-1], self.model, dataset.X_train_BC_Dir)
            else:
                self.tensorboard_writer.add_content(\
                        epoch, testing, self.training_losses.training_loss[-1],\
                        self.training_losses.training_loss_BC[-1], self.training_losses.training_loss_CP[-1],\
                        0, self.model, dataset.X_train_BC_Dir)
                    
            # save final model
            if (epoch == hyperparameter.numEpochs):
                self.tensorboard_writer.save_model(self.model)
                
                
        # passing data to tune 
        if self.ray_tune_study and dataset.X_test != None:
            tune.report(loss_test =  self.training_losses.testing_loss[-1], loss_train =  self.training_losses.training_loss[-1], \
                        loss_BC = self.training_losses.training_loss_BC[-1], loss_CP = self.training_losses.training_loss_CP[-1], run_number = self.tensorboard_writer.run_number)
                
            
        elif self.ray_tune_study and dataset.X_test == None:
            raise ValueError ("Ray tune study can only be executed if testing is activated")
          
            
        return None


    def __print_training_progress(self, testing, epoch):
        if testing:
            print("Epoch %d, PDE_loss: %.8f, ICBC_loss: %.8f, testing_loss: %.8f" % (
                epoch, self.training_losses.training_loss_CP[-1], self.training_losses.training_loss_BC[-1], self.training_losses.testing_loss[-1]))

        elif len(self.training_losses.training_loss_BC_Dir) != 0.:
            print("Epoch %d, PDE_loss: %.8f, Dir_BC_loss: %.8f, Neu_BC_loss: %.8f" % (
                epoch, self.training_losses.training_loss_CP[-1], self.training_losses.training_loss_BC_Dir[-1], self.training_losses.training_loss_BC_Neu[-1]))
            
        else:
            print("Epoch %d, PDE_loss: %.8f, ICBC_loss: %.8f" % (
                epoch, self.training_losses.training_loss_CP[-1], self.training_losses.training_loss_BC[-1]))
    
        return None
    



    
    
    
    