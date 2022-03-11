#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: philippbst
"""


import time
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bayesopt import BayesOptSearch
from dataclasses import dataclass


@dataclass()
class Dataset:
    X_train_CP: torch.tensor = None

    X_train_BC_Dir: torch.tensor = None
    U_train_BC_Dir: torch.tensor = None
    X_train_BC_Neu: torch.tensor = None
    Y_train_BC_Neu: torch.tensor = None
    X_train_BC_Robin: torch.tensor = None
    Y_train_BC_Robin: torch.tensor = None
    
    X_test: torch.tensor = None
    U_test: torch.tensor = None

@dataclass()
class Loss_curves:
    training_loss: list = None
    testing_loss: list = None
    
    training_loss_CP: list = None
    training_loss_BC: list = None
    
    training_loss_BC_Dir: list = None
    training_loss_BC_Neu: list = None
    training_loss_BC_Robin: list = None
   
@dataclass()
class Losses:
    loss_train: float = None
    loss_test: float = None
    
    loss_CP: float = None
    loss_BC: float = None
    
    loss_BC_Dir: float = None
    loss_BC_Neu: float = None
    loss_BC_Robin: float = None   
   

# Class that supports writing tensorboard files
class TensorboardWriterAssistant():
    def __init__(self, path_to_run_folders):
        self.path_to_run_folders = path_to_run_folders
        self.path_to_result_folder = None
        self.run_number = None
        self.writer = None
        self.running_epoch = 0

    # add the content during the training in each epoch
    def add_content(self, epoch, testing, loss, loss_ICBC, loss_CP, loss_test, model, X_train):
        self.running_epoch = self.running_epoch + 1
        if epoch == 1:
            self.writer.add_graph(model, X_train)

        # write losses to Tensorboard report
        self.writer.add_scalar('Loss/train_ges', loss, self.running_epoch)
        self.writer.add_scalar('Loss/train_BC', loss_ICBC, self.running_epoch)
        self.writer.add_scalar('Loss/train_CP', loss_CP, self.running_epoch)

        if testing:
            self.writer.add_scalar('Loss/test', loss_test, self.running_epoch)

    # create a summary writer for the soecific location
    def create_writer(self):
        if self.writer is None:
            next_num = self.__get_next_runfolder_number()
            next_run_folder = f"{self.path_to_run_folders}/run_{next_num}"
            self.path_to_result_folder = next_run_folder
            self.run_number = next_num
            self.writer = SummaryWriter(next_run_folder)
        else:
            raise RuntimeError("Writer was already created for this instance")
        
    # get the number for the next runfolder
    def __get_next_runfolder_number(self):
        run_numbers = []
        if len(glob(self.path_to_run_folders + '/*/')) == 0:
            run_numbers.append(0)
        else:
            for run_path in glob(self.path_to_run_folders + '/*/'):
                run_path = run_path[len(self.path_to_run_folders) + 1: len(run_path) - 1]
                name, number = run_path.split('_')
                latest_num = int(number)
                run_numbers.append(latest_num)

        max_run_number = max(run_numbers)
        next_num = max_run_number+1
        return next_num

    # safe the final model
    def save_model(self, model, ID = ""):
        model_name = "trained_model"
        save_path =  f"{self.path_to_result_folder}/{model_name}{ID}.pt"
        torch.save(model.state_dict(), save_path)
        


# class to summarize ray tune hyperparameter optimization setup
class RayTuneSetup():
    def __init__(self, training_function, config, path_to_studies):
        self.path_to_studies = path_to_studies
        self.training_function = training_function
        self.config = config
        
         
    def execute_analysis(self, num_cpus = 1):
        ray.shutdown()
        ray.init(num_cpus=num_cpus)
        reporter = CLIReporter(metric_columns=["loss_test", "loss_train", "loss_BC", "loss_CP", "run_number"])
        num = self.get_next_study_number()
        name_dir = f"study_{num}"
        
        analysis = tune.run(self.training_function, config = self.config, local_dir = self.path_to_studies,\
                            name = name_dir)
 
        '''
        progress_reporter = reporter
        max_failures = 3
        metric = "loss_test" 
        mode = "min"
        
        search_alg = bayesopt with  bayesopt = BayesOptSearch(config, metric = "loss_test", mode = "min")
        bayesopt = BayesOptSearch(self.config, metric = "loss_test", mode = "min")                   
        analysis = tune.run(self.training_function, local_dir = self.path_to_studies,\
                            name = name_dir, search_alg = bayesopt, config = {"num_samples": 10})
        '''
      
        ray.shutdown()
        return analysis

    def get_next_study_number(self):
        run_numbers = []
        if len(glob(self.path_to_studies + '/*/')) == 0:
            run_numbers.append(0)
        else:
            for run_path in glob(self.path_to_studies + '/*/'):
                run_path = run_path[len(self.path_to_studies) + 1: len(run_path) - 1]
                name, number = run_path.split('_')
                latest_num = int(number)
                run_numbers.append(latest_num)

        max_run_number = max(run_numbers)
        next_num = max_run_number+1
        return next_num
        
    
    
# can only handle methods with X as input
def check_gradientTracking(func):
    def inner(self, X):
        if not X.requires_grad:
            X.requires_grad = True
        return func(self, X)
    return inner


# can so far only handel the pinn training methods
def record_time(func):
    def inner(self, hyperparameter, dataset):
        tic = time.perf_counter()
        func(self, hyperparameter, dataset)
        toc = time.perf_counter()
        t_diff = abs(tic - toc)

        if t_diff < 60:
            print(f"\n >>> Elapsed time for Training:  {t_diff} s \n")
        elif (t_diff >= 60 and t_diff < 3600):
            t_min = int(np.floor(t_diff / 60))
            t_sec = t_diff - (60 * t_min)
            print(f"\n >>> Elapsed time for Training: {t_min} m {t_sec} s \n")

        elif t_diff >= 3600:
            t_h = int(np.floor(t_diff / 3600))
            t_min = int(np.floor((t_diff-(3600*t_h)) / 60))
            t_sec = t_diff - (60 * t_min)
            print(f"\n >>> Elapsed time for Training: {t_h} h {t_min} m {t_sec} s \n")

    return inner


# class to track the predictions of the model during the training for visualization purpose
class ProgressTracker():

    def __init__(self, epochStep, X):
        self.trackedEpochs = []
        self.trackedPredictions = []
        self.epochStep = epochStep
        self.evaluationPoints = X

    def track(self, epoch, model):
        X = torch.from_numpy(self.evaluationPoints).float()
        U_pred = model(X)
        self.trackedEpochs.append(epoch)
        self.trackedPredictions.append(U_pred.detach().numpy())

    def visualizePredictions(self, maxNumOfPlots):
        if maxNumOfPlots > len(self.trackedEpochs):
            maxNumOfPlots = len(self.trackedEpochs)

        numRecords = len(self.trackedEpochs)
        recordsToShow = np.ceil(np.linspace(0, numRecords-1, maxNumOfPlots))
        plotsPerRow = int(np.ceil(recordsToShow.size**0.5))
        plotRows = int(np.ceil(recordsToShow.size / plotsPerRow))

        fig, ax = plt.subplots(plotRows, plotsPerRow)

        count = 0
        for i in range(plotRows):
            for j in range(plotsPerRow):
                if count < maxNumOfPlots:
                    U_pred = self.trackedPredictions[int(recordsToShow[count])]
                    epoch = self.trackedEpochs[int(recordsToShow[count])]

                    u_pred = U_pred[:, 0]
                    v_pred = U_pred[:, 1]

                    x_deformed = self.evaluationPoints[:, 0] + u_pred
                    y_deformed = self.evaluationPoints[:, 1] + v_pred
                    total_displacement = ((u_pred)**2 + (v_pred)**2)**0.5

                    p = ax[i, j].scatter(x_deformed, y_deformed, c=total_displacement, cmap=cm.turbo)
                    ax[i, j].set_xlabel(r'$x [m]$')
                    ax[i, j].set_ylabel(r'$y [m]$')
                    ax[i, j].set_aspect("equal")
                    cbar = fig.colorbar(p, shrink=0.6, aspect=8, ax=ax[i, j], orientation='vertical')
                    ax[i, j].grid()
                    csfont = {'fontname':'Arial'}
                    ax[i, j].set_title(f"Prediction in epoch: {epoch}",**csfont)     
                    count = count + 1

    def visualizeDeviations(self, maxNumOfPlots, U):
        if maxNumOfPlots > len(self.trackedEpochs):
            maxNumOfPlots = len(self.trackedEpochs)

        numRecords = len(self.trackedEpochs)
        recordsToShow = np.ceil(np.linspace(0, numRecords-1, maxNumOfPlots))
        plotsPerRow = int(np.ceil(recordsToShow.size**0.5))
        plotRows = int(np.ceil(recordsToShow.size / plotsPerRow))

        fig, ax = plt.subplots(plotRows, plotsPerRow)

        u_true = U[:, 0]
        v_true = U[:, 1]

        count = 0
        for i in range(plotRows):
            for j in range(plotsPerRow):
                if count < maxNumOfPlots:
                    U_pred = self.trackedPredictions[int(recordsToShow[count])]
                    epoch = self.trackedEpochs[int(recordsToShow[count])]

                    u_pred = U_pred[:, 0]
                    v_pred = U_pred[:, 1]
                    dev = abs(u_pred - u_true) + abs(v_pred - v_true)
                    
                    p = ax[i, j].scatter(self.evaluationPoints[:, 0], self.evaluationPoints[:, 1], c=dev, cmap=cm.gist_heat)
                    ax[i, j].set_xlabel(r'$x [m]$')
                    ax[i, j].set_ylabel(r'$y [m]$')
                    ax[i, j].set_aspect("equal")
                    cbar = fig.colorbar(p, shrink=0.6, aspect=8, ax=ax[i, j], orientation='vertical')
                    cbar.ax.set_title(r'$\Delta$' + r'$u(x,y)$')
                    csfont = {'fontname':'Arial'}
                    ax[i, j].set_title(f"Deviation in epoch: {epoch}",**csfont)                    
                    ax[i, j].grid()
                    count = count + 1
                    
                    
             
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
