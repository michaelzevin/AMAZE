"""
Storm Colloms 21/6/23

Defines Class to instantiate and train noramlising flow for each channel used in inference.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.stats import entropy
from scipy.stats import norm, gaussian_kde
from scipy.special import logit

import copy
import torch
from  glasflow import RealNVP, CouplingNSF
from torch import nn
import wandb


class NFlow():

    #initialise flow with inputs, conditionals, including the type of network, real non-volume preserving,
    #or neural spline flow
    #spline flow increases the flexibility in the flow model
    def __init__(self, no_trans, no_neurons, no_blocks, no_bins, training_inputs, cond_inputs,
                batch_size, total_smdls, RNVP=False, device="cpu"):
                
        """
        Initialise Flow with inputed data, either RNVP or Spline flow.

        Parameters
        ----------
        no_trans : int
            number of transforms to give the flow
        no_neurons : int
            number of neurons of the flow per layer/transform
        no_blocks : int
            number of blocks
        training_inputs : int
            number of parameters in dataspace (binary parameters)
        cond_inputs : int
            number of population hyperparameters
        batch_size : int
            number of training and validation samples to use in each batch
        total_smdls : int
            total number of subpopulation models
        num_bins : int
            number of bins to use for a spline flow
        RNVP : bool
            whether or not to use realNVP flow, if False use spline
        device : str
            device on which to run pytorch operations, default is CPU, otherwise GPU enabled with device = 'cuda:0'
        """
        self.no_params = training_inputs
        self.batch_size = batch_size

        self.total_smdls = total_smdls
        self.cond_inputs = cond_inputs

        self.device = device # cuda:X where X is the slot of the GPU. run nvidia-smi in the terminal to see gpus

        if RNVP:
            self.network = RealNVP(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                    n_neurons = no_neurons, n_transforms = no_trans, n_blocks_per_transform = no_blocks,
                                    linear_transform = None, batch_norm_between_transforms=True)
        else:
            self.network = CouplingNSF(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                        n_neurons = no_neurons, n_transforms = no_trans,
                                        n_blocks_per_transform = no_blocks, batch_norm_between_transforms=True,
                                        linear_transform = None, num_bins=no_bins)

        self.network.to(device)

    def trainval(self, lr, epochs, batch_no, filename, training_data, val_data, use_wandb):
        """
        Train the normalising flow for the specified number of epochs using a set of training and validation data,
        and save the model with the best validation loss

        Parameters
        ----------
        lr : float
            the initial learning rate used to train the normalising flow, which is then reduced with cosine annealing
        epochs : int
            number iterations to train for  - 1 epoch goes through through entire dataset
        batch_no : int
            number of batches of data in one iteration
        filename : str
            directory and filename of where to save best flow model
        training_data : array
            set of training data points for the normlising flow
        val_data : array
            set of validation data points for the normlising flow
        use_wandb : bool
            If true, uses weights and biases to optimise neural network parameters
        """
        #set optimiser for flow, optimises flow parameters:
        #affine - s and t that shift and scale the transforms
        #spline - nodes used to model the distribution of CDFs
        optimiser = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=0)
        #set learning rate cheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=0, last_epoch=- 1, verbose=False)

        n_epochs = epochs
        n_batches = batch_no 

        #initialize best flow model
        best_epoch = 0
        best_val_loss = np.inf

        #record network values and outputs in dictionary as training
        self.history = {'train': [], 'val': [], 'lr': []}

        #training loop
        for n in range(n_epochs): 
            train_loss = 0
            unweighted_KL_train = 0

            #set flow into training mode
            self.network.train()
            self.history['lr'].append(scheduler.get_last_lr())
            
            #Training
            for _ in range(n_batches):
                #split training data into - train: binary params; conditional: pop hyperparams
                x_train, x_conditional, xweights = self.get_training_data(training_data)
                #sets flow optimisers gradients to zero
                optimiser.zero_grad()
                #calculate the training loss function for flow as -log_prob
                unweighted_KL = self.network.log_prob(x_train, conditional=x_conditional)
                loss = -(xweights*unweighted_KL).mean()
                #computes gradient of flow network parameters
                loss.backward()
                #steps optimtiser down gradient of loss surface
                optimiser.step()
                #track flow losses
                unweighted_KL_train += -unweighted_KL.mean().cpu().item()
                train_loss += loss.cpu().item()
            scheduler.step()

            #track and average losses
            train_loss /= n_batches
            self.history['train'].append(train_loss)
            
            # Validate
            with torch.no_grad(): #disables gradient caluclation
                #call validation data
                x_val, x_conditional_val, x_weights_val = self.get_val_data(val_data)
                
                #evaluate flow parameters
                self.network.eval()

                #calculate flow validation loss
                unweighted_KL_loss = self.network.log_prob(x_val, conditional=x_conditional_val)
                val_loss = - (unweighted_KL_loss).mean()
                val_loss = - (x_weights_val*unweighted_KL_loss).mean()
                total_val_loss=val_loss.cpu().numpy() 
                total_unweighted_KL_val = -unweighted_KL_loss.mean().cpu().numpy()
                #save the loss value of the training data
                self.history['val'].append(total_val_loss)

            #print history
            sys.stdout.write(
                    '\r Epoch: {} || Training loss: {} || Validation loss: {}'.format(
                    n+1, train_loss, total_val_loss))
            
            #track losses for weights and biases
            if use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": total_val_loss, "unweighted_train_KL": unweighted_KL_train, "unweighted_val_KL": total_unweighted_KL_val})

            #copy the best flow model
            if total_val_loss < best_val_loss:
                best_epoch = n
                best_val_loss = total_val_loss
                best_model = copy.deepcopy(self.network.state_dict())

        #save best model
        print(f'\n Best epoch: {best_epoch}')
        self.network.load_state_dict(best_model)
        torch.save(best_model, f'{filename}.pt')
        self.plot_history(filename)

    def plot_history(self,filename):
        """
        Plots losses for training of network

        filename : str
            directory and filename of where to save loss data and figures
            alongside best flow model
        """

        #loss plot
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(self.history['val'][5:], label = 'Val loss', color='tab:orange')
        ax.plot(self.history['train'][5:], label = 'Train loss', color='tab:blue')
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_xlabel('Epochs', fontsize=10)
        ax.tick_params(axis='both', labelsize=10)
        text = ax.yaxis.get_offset_text()
        text.set_size(10)
        ax.legend(loc = 'lower left', prop={'size':10})

        #inset log plot
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        valloss = np.asarray(self.history['val'][1:])
        trainloss = np.asarray(self.history['train'][1:])
        axins.plot(valloss, color='tab:orange')
        axins.plot(trainloss, color='tab:blue')
        axins.set_xscale('log')
        axins.tick_params(axis='both', labelsize=10)
        text = axins.yaxis.get_offset_text()
        text.set_size(10)

        #save loss data
        plt.savefig(f'{filename}loss.pdf')
        pd.DataFrame.to_csv(pd.DataFrame.from_dict(self.history),f'{filename}_loss_history.csv')

        #plot learning rate
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(self.history['lr'], label = 'lr')
        ax.set_ylabel('Learning rate', fontsize=10)
        ax.set_xlabel('Epochs', fontsize=10)
        plt.savefig(f'{filename}lr.pdf')



    def sample(self, conditional,no_samples):
        """
        Pull samples from flow given one pair of population hyperparameters.

        Parameters
        ----------
        coditional : array
            list/array of hyperparameter values to use as conditional values from which to sample from the flow, 
            of shape [Nhyperparams]
        no_samples : int
            number of samples to take for each conditional
        
        Returns
        -------
        samples : array 
            Flow samples in the logistically-mapped space of shape [no_samples, Nparams]
        """
        samples = np.zeros((no_samples, self.no_params))

        #check that requested conditional are the same shape as Nhyperparameters
        if conditional.astype(np.float32).shape != self.cond_inputs:
            raise ValueError(f"Expected shape {self.cond_inputs} but got {conditional.astype(np.float32).shape}")

        with torch.no_grad():
            conditional = torch.from_numpy(conditional.astype(np.float32))
            #tile as many conditional hyperparameter values as no samples
            conditional = conditional.tile(no_samples,1)
            samples = self.network.sample(no_samples, conditional=conditional)

        return(samples)

    def get_training_data(self, training_samples):
        """
        Get random batch training data from training_samples
        
        Returns
        -------
        xdata : tensor 
            a batch of training data samples of shape [no_samples, self.no_params]
        xhyperparams : tensor
            the corersponding conditional hyperparameters to the batch of training data
            of shape [no_samples, self.cond_inputs]
        xweights : tensor
            the corersponding sample weigths to the batch of training data
            of shape [no_samples]
        """
        #retrieve random samples of size batch_size from training samples
        random_sample_idxs = np.random.choice(np.shape(training_samples)[0],size=(int(self.batch_size)))

        #retrieve dataspace samples, and corresponding hyperparameters and weights to the randomly drawn samples
        batched_samples = training_samples[random_sample_idxs,:self.no_params]
        batched_hp_pairs = training_samples[random_sample_idxs, self.no_params:self.cond_inputs]
        batch_weights = training_samples[random_sample_idxs,-1]

        #reshape tensors to be correct shape
        xdata=torch.from_numpy(batched_samples.astype(np.float32)).to(self.device)
        xhyperparams = torch.from_numpy(batched_hp_pairs.astype(np.float32)).to(self.device)
        xhyperparams = xhyperparams.reshape(-1,self.cond_inputs)
        xweights = torch.from_numpy(batch_weights.astype(np.float32)).to(self.device)

        return(xdata, xhyperparams,xweights)

    def get_val_data(self, validation_data):
        """
        Get random batch validation data from self.validation_data
        
        Returns
        -------
        xval : tensor 
            a batch of validation data samples of shape [no_samples, self.no_params]
        xhyperparams : tensor
            the corersponding conditional hyperparameters to the batch of validation data
            of shape [no_samples, self.cond_inputs]
        xweights : tensor
            the corersponding sample weigths to the batch of validation data
            of shape [no_samples]
        """

        #pull batch from data
        random_samples = np.random.choice(np.shape(validation_data)[0], size=(int(self.batch_size)))

        #retrieve dataspace samples, and corresponding hyperparameters and weights to the randomly drawn samples
        batched_samples = validation_data[random_sample_idxs,:self.no_params]
        batched_hp_pairs = validation_data[random_sample_idxs, self.no_params:self.cond_inputs]
        batch_weights = validation_data[random_sample_idxs,-1]

        #reshape
        xval=torch.from_numpy(validation_samples.astype(np.float32)).to(self.device)
        xhyperparams = torch.from_numpy(validation_hp_pairs.astype(np.float32)).to(self.device)
        xhyperparams = xhyperparams.reshape(-1,self.cond_inputs)
        xweights = torch.from_numpy(val_weights.astype(np.float32)).to(self.device)
        return(xval, xhyperparams, xweights)

    def load_model(self,filename):
        """
        Load pre-trained flow from saved model, and set flow to evaluation mode
        """
        self.network.load_state_dict(torch.load(filename, map_location=torch.device(self.device)))
        self.network.eval()

    def log_jacobian(self,sample, mappings):
        """
        Calculate the log jacobian term to add to the log likelihood to account for the logistic transforms
        of the samples of [mchirp, q, chieff, z]

        returns the sum of the log of the absolute value of the jacobian term
        """
        #dtheta prime by dtheta
        jac = torch.zeros(sample.shape[0], self.no_params).to(self.device)

        #loop over number of params and add jacobian term, assuming all dimensions have undergone a logistic mapping
        for i in range(self.no_params):
            jac[:,i] = mappings[i+1]/((sample[:,i])*(mappings[i+1]-(sample[:,i]))*mappings[i])
        
        return torch.sum(torch.log(torch.abs(jac)), dim=1)

    def get_logprob(self, sample, mapped_sample, mappings, conditionals):
        """
        get log_prob p(theta|Lambda) given a sample of gw observables theta given conditional hyperparameters Lambda

        Parameters
        ----------
        sample : array
            posterior samples of GW observations in unmapped data-space
            [Nobs x Nsamples x Nparams] shape array
        mapped_sample : array
            posterior samples mapped into logistic space with Nflow.map_obs function
            [Nobs x Nsamples x Nparams] shape array
        conditionals : array
            values of population hyperparameters
            [Nobs x Nsamples x Nconditionals] shapped array

        Returns
        ----------
        log_prob : array
            the log probability of each sample
            [Nobs x Nsamples] shaped array
        """

        #make sure samples in right format
        sample = torch.from_numpy(sample.astype(np.float32)).to(self.device)
        mapped_sample = torch.from_numpy(mapped_sample.astype(np.float32)).to(self.device)
        hyperparams = torch.from_numpy(conditionals.astype(np.float32)).to(self.device)
        #store shape
        shape = mapped_sample.shape

        #flatten samples given they are have dimensions Nsamples x Nobs x Nparams
        sample = torch.flatten(sample, start_dim=0, end_dim=1)
        mapped_sample = torch.flatten(mapped_sample, start_dim=0, end_dim=1)
        hyperparams = torch.flatten(hyperparams, start_dim=0, end_dim=1)
        hyperparams = hyperparams.reshape(-1,self.cond_inputs)
        sample = sample.reshape(-1,self.no_params)
        mapped_sample = mapped_sample.reshape(-1,self.no_params)

        with torch.no_grad():
            log_prob = self.network.log_prob(mapped_sample, hyperparams)
            log_prob += self.log_jacobian(sample, mappings)

            #reshape
            log_prob = torch.reshape(log_prob, [shape[0],shape[1]])

            log_prob = log_prob.cpu().numpy() 
            if np.any(np.isnan(log_prob)):
                sys.exit('Unexpected nans in log prob evaluation, quitting code.')
                log_prob[np.isnan(log_prob)] = -np.inf

        return log_prob

    def get_latent_samps(self, samps, conditionals):
        """
        Returns samps of shape [Nsamps, Nparams] mapped to the latent space conditional on conditionals of shape [Nsamps, Nconditionals]
        """
        
        samps = torch.from_numpy(samps.astype(np.float32))
        conditionals = torch.from_numpy(conditionals.astype(np.float32))

        latent_samples, _= self.network.forward(samps.reshape(-1,self.no_params), conditional=conditionals.reshape(-1,self.cond_inputs))
        return latent_samples