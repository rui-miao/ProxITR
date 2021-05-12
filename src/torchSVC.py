# Copyright (c) Rui Miao 2021
# Licensed under the MIT License.
#%%
import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_sample_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import RandomSampler, BatchSampler

#%%
class RegimeLearner(BaseEstimator):
    def __init__(self, batch_size=100, rho=0.1, n_epoch=100, learning_rate=1e-1, device='cuda', opt='LBFGS'):
        self.rho = rho
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.device = device #cpu or cuda
        self.opt = opt # 'LBFGS' or 'SGD'
        return

    def train_LBFGS(self, X, y, sample_weight, model, n_epoch=5000):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device, requires_grad=False)
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=self.device)

        optimizer = optim.LBFGS(model.parameters(), lr=self.learning_rate, max_iter=2000, line_search_fn='strong_wolfe')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                        factor=0.1, patience=3, threshold=1e-3)

        def loss_fun(output):
            return torch.mean(sample_weight\
                                * F.smooth_l1_loss(torch.clamp(1 - y*output, min=0),
                                                    torch.zeros_like(y, dtype=torch.float32, device=self.device),
                                                    reduction='none', beta = 0.01))

        # Set model to training mode
        model.train()

        # Compute initials
        optimizer.zero_grad()
        output = model(X).squeeze()
        weight_old = model.weight.squeeze()
        bias_old   = model.bias.squeeze()
        loss_old = loss_fun(output) + self.rho * (torch.dot(weight_old, weight_old)) / 2.

        for epoch in range(n_epoch):

            def closure():
                optimizer.zero_grad()
                output = model(X).squeeze()
                weight = model.weight.squeeze()
                loss_all = loss_fun(output)
                loss_all += self.rho * (torch.dot(weight, weight)) / 2.
                loss_all.backward()
                return loss_all

            optimizer.step(closure)
            output = model(X).squeeze()
            weight_new = model.weight.squeeze()
            bias_new   = model.bias.squeeze()
            loss_new = loss_fun(output) + self.rho * (torch.dot(weight_new, weight_new)) / 2.

            x_dist = (torch.norm(weight_new - weight_old)+torch.norm(bias_new - bias_old))/(torch.norm(weight_old)+torch.norm(bias_old))
            loss_dist = torch.abs(loss_new - loss_old)/torch.max(torch.tensor(1, dtype=torch.float32, device=self.device), torch.abs(loss_old))

            scheduler.step(loss_new)

            # Early stopping using scheduler's learning_rate
            # print("Epoch: {:4d}\tloss: {}".format(epoch, loss_all / N), 'lr=', scheduler._last_lr)
            if (float(scheduler._last_lr[-1]) < 2e-10  or  x_dist < 1e-5  or  loss_dist < 1e-8):
                break

            weight_old = weight_new.clone()
            bias_old   = bias_new.clone()
            loss_old   = loss_new.clone()


    def train_SGD(self, X, y, sample_weight, model, batch_size, n_epoch=5000):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device, requires_grad=False)
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=self.device)
        N = len(sample_weight)

        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.rho)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                        factor=0.1, patience=3, threshold=1e-3)

        def loss_fun(output, y, sample_weight):
            return torch.mean(sample_weight\
                                * F.smooth_l1_loss(torch.clamp(1 - y*output, min=0),
                                                    torch.zeros_like(y, dtype=torch.float32, device=self.device),
                                                    reduction='none', beta = 1e-3))

        # Set model to training mode
        model.train()

        for epoch in range(n_epoch):
            for ind in BatchSampler(RandomSampler(range(N)), batch_size=batch_size, drop_last=True):
                X_samp             = X[ind]
                y_samp             = y[ind]
                sample_weight_samp = sample_weight[ind]

                optimizer.zero_grad()
                output = model(X_samp).squeeze()
                #weight = model.weight.squeeze()

                loss = loss_fun(output, y_samp, sample_weight_samp)

                loss.backward()
                optimizer.step()

            # calculate val loss and apply scheduler
            output = model(X).squeeze()
            weight = model.weight.squeeze()
            loss_all = loss_fun(output, y, sample_weight)
            loss_all += self.rho * (torch.dot(weight, weight)) / 2. #note: AdamW's l2 weight decay is not in loss_all
            scheduler.step(loss_all)

            # Early stopping using scheduler's learning_rate
            # print("Epoch: {:4d}\tloss: {}".format(epoch, loss_all / N), 'lr=', scheduler._last_lr)
            if float(scheduler._last_lr[-1]) < 2e-8:
                break

    
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=True)
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = _check_sample_weight(sample_weight, X)
        # Check cuda GPU device
        if self.device == 'cuda':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        
        self.model = nn.Linear(in_features=X.shape[1], out_features=1, bias=True)
        self.model.to(device=self.device)
        if self.opt == 'LBFGS':
            self.train_LBFGS(X=X, y=y, sample_weight=sample_weight, model=self.model, n_epoch=self.n_epoch)
        elif self.opt == 'SGD':
            self.train_SGD(X=X, y=y, sample_weight=sample_weight, model=self.model, batch_size=self.batch_size, n_epoch=self.n_epoch)

        # Coefficients
        self.W = self.model.weight.squeeze().detach().cpu().numpy()
        self.b = self.model.bias.squeeze().detach().cpu().numpy()

        return self
    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        # Set model to evaluation mode
        self.model.eval()
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass
            #r = torch.sign(self.model(X))
            r = self.model(X)
        r = r.detach().cpu().numpy().reshape(-1)
        return r

    def score(self, X, y):
        pass
