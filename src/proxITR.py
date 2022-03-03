# Copyright (c) Rui Miao 2021-2022
# Licensed under the MIT License.
#%%
import numpy as np
import pandas as pd
from src.rkhs_scaler import RKHSIV, RKHSIVCV, ApproxRKHSIV, ApproxRKHSIVCV, RKHSIV_q, RKHSIVCV_q, ApproxRKHSIV_q, ApproxRKHSIVCV_q
import sklearn
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn import pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import RobustScaler as Scaler
#from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from src.torchSVC import RegimeLearner
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats

#from copy import deepcopy

def _check_all(param):
    return (isinstance(param, str) and (param == 'all'))


#%%
class ApproxNonLinear:
    def __init__(self, transX, featX, rho = 0.1, n_epoch = 2000, batch_size = 200, learning_rate=0.1, opt = 'LBFGS'):
        """
        Parameters:
            rho : the penalty coefficient
        """
        self.transX = transX  # a fitted transformer
        self.featX = featX      # Nystroem features
        self.rho = rho
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt = opt
    
    def fit(self, X, y, sample_weight):
        # Standardize X
        X = self.transX.transform(X)
        # Kernel Approximation
        X_transformed = self.featX.transform(X)
        self.regime = RegimeLearner(n_epoch=self.n_epoch, batch_size = self.batch_size, rho = self.rho, learning_rate=self.learning_rate, opt=self.opt)\
            .fit(X=X_transformed, y=y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        X_transformed = self.featX.transform(X)
        return self.regime.predict(X=X_transformed)

def FastOptBW(X, y, sample_weight=None, n_gammas=10):
    """
    Fast Optimal Bandwidth Selection for RBF Kernel using HSIC (Damodaran 2018, IEEE)
    Input: 
        X: covariates of SVM
        y: response of SVM: {-1,1}
        sample_weight: for weight w_i on sample (X_i,y_i), a posiitive np vector with sum=1
        n_gammas: number of gammas to try
    Output: 
        Optimal gamma for the RBF kernel in terms of classification
    """
    # Check cuda GPU device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    X = torch.tensor(X, dtype=torch.float32, device=device)
    K_X_euclidean = torch.square(torch.cdist(X,X))
    triuInd = torch.triu_indices(K_X_euclidean.size(0),K_X_euclidean.size(0),offset=1)
    K_X_euclidean_upper = K_X_euclidean[triuInd[0],triuInd[1]]
    gammas = 1./torch.quantile(K_X_euclidean_upper,
                                torch.linspace(0.1, 0.9, n_gammas, device=device))

    reci_pos = 2./(y.shape[0] + np.sum(y))
    reci_neg = 2./(y.shape[0] - np.sum(y))
    K_y = torch.zeros_like(K_X_euclidean)
    K_y[y>0,y>0]=reci_pos
    K_y[y<0,y<0]=reci_neg

    if sample_weight is None:
        H = torch.eye(X.size(0), dtype=torch.float32, device=device) - torch.ones_like(K_X_euclidean)/X.size(0) 
    else:
        if np.any(sample_weight<0):
            raise ValueError('Negative weight detected!')
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=device)
        sample_weight /= sample_weight.sum()
        H = torch.diag(sample_weight) - torch.outer(sample_weight, sample_weight)
    HK_yH = H@K_y@H

    HSICs = torch.zeros(n_gammas, device=device)
    K_X = torch.zeros_like(K_X_euclidean)
    for it, gamma in enumerate(gammas):
        torch.exp(-gamma*K_X_euclidean, out=K_X)
        HSICs[it] = torch.sum(K_X*HK_yH)

    return gammas[torch.argmax(HSICs)].data.tolist()
#%%
class proxITR:
    def __init__(self, A,X,Z,W,Y, god=False, U=None, h0=None, q0=None, d_X_opt=None, learning_rate=0.1, n_epoch=10000, batch_size = 200, n_components = 150, opt='LBFGS', verbose=False):
        self.A = A.to_numpy(dtype=int).reshape(-1)
        self.X = X.to_numpy()
        self.Z = Z.to_numpy()
        self.W = W.to_numpy()
        self.Y = Y.to_numpy().reshape(-1)
        self.god=god
        if self.god:
            try:
                self.d_X_opt = d_X_opt.to_numpy(dtype=int).reshape(-1)
                self.U = U.to_numpy()
                self.h0 = h0.to_numpy()
                self.q0 = q0.to_numpy()
            except:
                pass
        # self.max_iter=10000
        # self.tol=1e-5
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.PipeNonLinear = pipeline.Pipeline([("Scaler", Scaler()),
                                  ("feature_map", 
                                        Nystroem(kernel="rbf", n_components=150, random_state=None)),
                                  ("svm",
                                        RegimeLearner(batch_size=self.batch_size, n_epoch=self.n_epoch))])
        self.PipeLinear = pipeline.Pipeline([("Scaler", Scaler()),
                                  ("svm", 
                                        RegimeLearner(batch_size=self.batch_size, n_epoch=self.n_epoch))])
        self.opt = opt # 'LBFGS' or 'SGD'
        self.batch_size = batch_size
        self.n_components = n_components
        self.verbose = verbose
        

    def fit_h0_a0(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_h0_a0.predict can be used to calculate h0(W,0,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill

        RKHS_h0_a0 = ApproxRKHSIV(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale)\
                 .fit(np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])
        return RKHS_h0_a0.predict
    def fit_h0_a0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales = 'auto', n_components=20, cv=5, index='all'):
        """
        RKHS_h0_a0.predict can be used to calculate h0(W,0,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
            index_all=True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill
            index_all=False

        RKHS_h0_a0 = ApproxRKHSIVCV(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_h0_a0     = RKHS_h0_a0.gamma_f
            self.gamma_h_h0_a0     = RKHS_h0_a0.gamma_h
            self.alpha_scale_h0_a0 = RKHS_h0_a0.best_alpha_scale
            self.predict_h0_a0     = RKHS_h0_a0.predict
            self.cv_h0_a0          = cv

        return RKHS_h0_a0.gamma_f, RKHS_h0_a0.gamma_h, RKHS_h0_a0.best_alpha_scale, RKHS_h0_a0.predict

    def fit_h0_a1(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_h0_a1.predict can be used to calculate h0(W,1,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = self.A.astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill

        RKHS_h0_a1 = ApproxRKHSIV(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale)\
                 .fit(np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])
        return RKHS_h0_a1.predict
    def fit_h0_a1_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales = 'auto', n_components=20, cv=5, index='all'):
        """
        RKHS_h0_a1.predict can be used to calculate h0(W,1,X)
        index is rows of data: if ='all', then use all data, otherwise is a 1 dim np.array
        """
        if _check_all(index): 
            index = self.A.astype(bool)
            index_all=True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill
            index_all=False

        RKHS_h0_a1 = ApproxRKHSIVCV(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.W,self.X),axis=1)[index,:],
                      self.Y[index],
                      np.concatenate((self.Z,self.X),axis=1)[index,:])

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_h0_a1     = RKHS_h0_a1.gamma_f
            self.gamma_h_h0_a1     = RKHS_h0_a1.gamma_h
            self.alpha_scale_h0_a1 = RKHS_h0_a1.best_alpha_scale
            self.predict_h0_a1     = RKHS_h0_a1.predict
            self.cv_h0_a1          = cv

        return RKHS_h0_a1.gamma_f, RKHS_h0_a1.gamma_h, RKHS_h0_a1.best_alpha_scale, RKHS_h0_a1.predict

    def predict_h0(self, n_components=20, gamma_f='auto', gamma_h=0.1):
        """
        Evaluate the performance of fit_h0_a0 and fit_h0_a1 using all data
        """
        self.h0_est = np.ndarray(self.A.shape[0])
        predict_h0_a0 = self.fit_h0_a0(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h)
        predict_h0_a1 = self.fit_h0_a1(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h)
        self.h0_est[(1-self.A).astype(bool)] = predict_h0_a0(np.concatenate((self.W,self.X),axis=1)[(1-self.A).astype(bool),:])
        self.h0_est[self.A.astype(bool)]     = predict_h0_a1(np.concatenate((self.W,self.X),axis=1)[self.A.astype(bool),:])
        #self.RKHS_h0.predict(np.concatenate((self.W.reshape(-1,1),self.A.reshape(-1,1),self.X),axis=1))
        if self.god:
            print("RMSE of h0_hat: ",sklearn.metrics.mean_squared_error(self.h0, self.h0_est, squared=False), '\n')
        return
    def predict_h0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales, n_components, cv):
        """
        Evaluate the performance of fit_h0_a0 and fit_h0_a1 using all data
        """
        self.h0_est = np.ndarray(self.A.shape[0])
        #predict_h0_a0 = self.fit_h0_a0(kernel=kernel, gamma=gamma, delta_scale=delta_scale, delta_exp=delta_exp, alpha_scale=alpha_scale)
        #predict_h0_a1 = self.fit_h0_a1(kernel=kernel, gamma=gamma, delta_scale=delta_scale, delta_exp=delta_exp, alpha_scale=alpha_scale)
        _,_,_,predict_h0_a0 = self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv, index='all')
        _,_,_,predict_h0_a1 = self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv, index='all')
        self.h0_est[(1-self.A).astype(bool)] = predict_h0_a0(np.concatenate((self.W,self.X),axis=1)[(1-self.A).astype(bool),:])
        self.h0_est[self.A.astype(bool)]     = predict_h0_a1(np.concatenate((self.W,self.X),axis=1)[self.A.astype(bool),:])
        #self.RKHS_h0.predict(np.concatenate((self.W.reshape(-1,1),self.A.reshape(-1,1),self.X),axis=1))
        if self.god:
            print("RMSE of h0_hat: ",sklearn.metrics.mean_squared_error(self.h0, self.h0_est, squared=False), '\n')
        return

    def Delta(self, n_components=20, gamma_f='auto', gamma_h=0.1):
        """
        Calculate Delta_est using all data
        """
        predict_h0_a0 = self.fit_h0_a0(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h)
        predict_h0_a1 = self.fit_h0_a1(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h)

        self.Delta_est = predict_h0_a1(np.concatenate((self.W,self.X),axis=1))\
                         - predict_h0_a0(np.concatenate((self.W,self.X),axis=1))
        return
    def Delta_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales, n_components=20, cv=5):
        """
        Calculate Delta_est using all data
        """
        gamma_f_h0_a0, gamma_h_h0_a0, alpha_scale_h0_a0, predict_h0_a0 \
            = self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv)
        gamma_f_h0_a1, gamma_h_h0_a1, alpha_scale_h0_a1, predict_h0_a1 \
            = self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv)

        self.Delta_est = predict_h0_a1(np.concatenate((self.W,self.X),axis=1))\
                         - predict_h0_a0(np.concatenate((self.W,self.X),axis=1))
        return

    def fit_q0_a0(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_q0_a0.predict can be used to calculate q(Z,0,X) =>q0(Z,X)
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill

        RKHS_q0_a0 = ApproxRKHSIV_q(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1),
                      index)
        
        return RKHS_q0_a0.predict
    def fit_q0_a0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales='auto', n_components=20, cv=5, index='all'):
        """
        RKHS_q0_a0.predict can be used to calculate q(Z,0,X) =>q0(Z,X)
        """
        if _check_all(index): 
            index = (1-self.A).astype(bool)
            index_all=True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = (1-self.A).astype(bool) & fill
            index_all=False

        RKHS_q0_a0 = ApproxRKHSIVCV_q(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1), 
                      index)

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_q0_a0     = RKHS_q0_a0.gamma_f
            self.gamma_h_q0_a0     = RKHS_q0_a0.gamma_h
            self.alpha_scale_q0_a0 = RKHS_q0_a0.best_alpha_scale
            self.predict_q0_a0     = RKHS_q0_a0.predict
            self.cv_q0_a0          = cv
        
        return RKHS_q0_a0.gamma_f, RKHS_q0_a0.gamma_h, RKHS_q0_a0.best_alpha_scale, RKHS_q0_a0.predict

    def fit_q0_a1(self, n_components=20, gamma_f='auto', gamma_h=0.1, alpha_scale='auto',index='all'):
        """
        RKHS_q0_a1.predict can be used to calculate q(Z,1,X) =>q1(Z,X)
        """
        if _check_all(index): 
            index = self.A.astype(bool)
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill

        RKHS_q0_a1 = ApproxRKHSIV_q(n_components=n_components, gamma_f=gamma_f, gamma_h=gamma_h, alpha_scale=alpha_scale).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1), 
                      index)

        return RKHS_q0_a1.predict 
    def fit_q0_a1_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales='auto', n_components=20, cv=5, index='all'):
        """
        RKHS_q0_a1.predict can be used to calculate q(Z,1,X) =>q1(Z,X)
        """
        if _check_all(index): 
            index = self.A.astype(bool)
            index_all = True
        else:
            fill = np.zeros_like(self.A, dtype=bool)
            fill[index]=True
            index = self.A.astype(bool) & fill
            index_all = False

        RKHS_q0_a1 = ApproxRKHSIVCV_q(gamma_f=gamma_f,n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, cv=cv, n_components=n_components).fit(
                      np.concatenate((self.Z, self.X),axis=1), 
                      np.ones_like(index), 
                      np.concatenate((self.X, self.W),axis=1), 
                      index)

        # save tuned parameters and predictor to self if index='all'
        if index_all: 
            self.gamma_f_q0_a1     = RKHS_q0_a1.gamma_f
            self.gamma_h_q0_a1     = RKHS_q0_a1.gamma_h
            self.alpha_scale_q0_a1 = RKHS_q0_a1.best_alpha_scale
            self.predict_q0_a1     = RKHS_q0_a1.predict
            self.cv_q0_a1          = cv

        return RKHS_q0_a1.gamma_f, RKHS_q0_a1.gamma_h, RKHS_q0_a1.best_alpha_scale, RKHS_q0_a1.predict

    def predict_q0(self, n_components, index):
        """
        q0_est : calculate estimate of q0(Z_i, A_i, L_i) on data[index,:]
        """

        q0_est = np.ndarray(self.A.shape[0])
        predict_q0_a0 = self.fit_q0_a0(n_components=n_components, gamma_f=self.gamma_f_q0_a0, gamma_h=self.gamma_h_q0_a0, alpha_scale = self.alpha_scale_q0_a0, index = index)
        predict_q0_a1 = self.fit_q0_a1(n_components=n_components, gamma_f=self.gamma_f_q0_a1, gamma_h=self.gamma_h_q0_a1, alpha_scale = self.alpha_scale_q0_a1, index = index)
        q0_est[(1-self.A).astype(bool)] = predict_q0_a0(np.concatenate((self.Z, self.X),axis=1)[(1-self.A).astype(bool),:])
        q0_est[self.A.astype(bool)]     = predict_q0_a1(np.concatenate((self.Z, self.X),axis=1)[self.A.astype(bool),:])

        return q0_est[index]
    def predict_q0_cv(self, gamma_f, n_gamma_hs, n_alphas, alpha_scales='auto', n_components=20, cv=5):
        """
        self.q0_est : calculate estimate of q0(Z_i, A_i, L_i) based on all data
        """
        self.q0_est = np.ndarray(self.A.shape[0])
        # _,_,_,predict_q0_a0 = self.fit_q0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, n_components=n_components, cv=cv)
        # _,_,_,predict_q0_a1 = self.fit_q0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, n_components=n_components, cv=cv)
        self.fit_q0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv)
        self.fit_q0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv)
        self.q0_est[(1-self.A).astype(bool)] = self.predict_q0_a0(np.concatenate((self.Z, self.X),axis=1)[(1-self.A).astype(bool),:])
        self.q0_est[self.A.astype(bool)]     = self.predict_q0_a1(np.concatenate((self.Z, self.X),axis=1)[self.A.astype(bool),:])
        # if self.god:
        #     print("RMSE of q0_est: ",sklearn.metrics.mean_squared_error(self.q0, self.q0_est, squared=False), '\n')
        return

    def fit_d1_X_cv(self, gamma_f, n_gamma_hs, n_alphas=20, alpha_scales='auto', n_components=50, cv_n=5, cv_r=5, linearity = 'nonlinear', rhos = [0.005, 0.01, 0.05, 0.1, 0.5], learning_rate=None):
        if learning_rate is None:
            learning_rate=self.learning_rate
        # if (not 'gamma_f_h0_a0' in self.__dict__) or (not 'gamma_h_h0_a0' in self.__dict__) or (not 'alpha_scale_h0_a0' in self.__dict__) or (not 'predict_h0_a0' in self.__dict__) or (cv_n != self.cv_h0_a0):
        #     self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        # if (not 'gamma_f_h0_a1' in self.__dict__) or (not 'gamma_h_h0_a1' in self.__dict__) or (not 'alpha_scale_h0_a1' in self.__dict__) or (not 'predict_h0_a1' in self.__dict__) or (cv_n != self.cv_h0_a1):
        #     self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)

        data_WX = np.concatenate((self.W,self.X),axis=1)
        Delta_est = self.predict_h0_a1(data_WX) - self.predict_h0_a0(data_WX)
        Delta_sign = np.sign(Delta_est)
        Delta_abs  = np.abs(Delta_est)
        if linearity=='nonlinear':
            transX = Scaler()
            X = transX.fit_transform(self.X)
            gamma_SVC = FastOptBW(X = X, y = Delta_sign, sample_weight=Delta_abs)
            featX = Nystroem(kernel='rbf', gamma=gamma_SVC, random_state=1, n_components=self.n_components).fit(X)

        if (type(rhos) is not float) and (len(rhos)>1):
            values = []
            for it, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X,Delta_sign)):
                values.append([])
                predict_h0_a0_train = self.fit_h0_a0(n_components=n_components, gamma_f=self.gamma_f_h0_a0, gamma_h=self.gamma_h_h0_a0, alpha_scale=self.alpha_scale_h0_a0, index=train_index)
                predict_h0_a1_train = self.fit_h0_a1(n_components=n_components, gamma_f=self.gamma_f_h0_a1, gamma_h=self.gamma_h_h0_a1, alpha_scale=self.alpha_scale_h0_a1, index=train_index)
                Delta_est_train = predict_h0_a1_train(data_WX[train_index,:]) - predict_h0_a0_train(data_WX[train_index,:])
                Delta_sign_train = np.sign(Delta_est_train)
                Delta_abs_train  = np.abs(Delta_est_train)
                for rho in rhos:
                    # d1_X = SVC(kernel='rbf', C=rho, gamma=gamma_SVC) # original SVM with kernel rbf
                    if linearity=='nonlinear':
                        d1_X = ApproxNonLinear(transX=transX, featX=featX, rho=rho, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
                        d1_X.fit(X = self.X[train_index,:], 
                                    y = Delta_sign_train,
                                    sample_weight=Delta_abs_train/Delta_abs_train.mean())
                    if linearity=='linear':
                        d1_X = clone(self.PipeLinear)
                        d1_X.set_params(svm__rho=rho, svm__learning_rate=learning_rate)
                        d1_X.fit(X = self.X[train_index,:], 
                                y = Delta_sign_train,
                                svm__sample_weight=Delta_abs_train/Delta_abs_train.mean())
                    if linearity=='tree':
                        d1_X = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho)
                        d1_X.fit(X = self.X[train_index,:],
                                y = Delta_sign_train,
                                sample_weight=Delta_abs_train/Delta_abs_train.mean())

                    d1_X_est = np.sign(d1_X.predict(self.X[test_index,:]))
                    h0_test = np.zeros_like(d1_X_est)
                    if np.sum(d1_X_est==-1)>0:
                        #h0_test[d1_X_est==-1] = predict_h0_a0_train(data_WX[test_index,:][d1_X_est==-1,:])
                        h0_test[d1_X_est==-1] = self.predict_h0_a0(data_WX[test_index,:][d1_X_est==-1,:])
                    if np.sum(d1_X_est==1)>0:
                        #h0_test[d1_X_est==1]  = predict_h0_a1_train(data_WX[test_index,:][d1_X_est==1,:])
                        h0_test[d1_X_est==1]  = self.predict_h0_a1(data_WX[test_index,:][d1_X_est==1,:])
                    values[it].append(np.mean(h0_test))

            avg_values = np.mean(np.array(values), axis=0)
            rho_best = rhos[np.argmax(avg_values)]

            if self.verbose:
                print("d1_X")
                print("rho: ",rhos)
                print("value: ",avg_values)
                print("rho_best:", rho_best)
        else:
            rho_best = rhos

        #d1_X = SVC(kernel='rbf', C=rho_best, gamma=gamma_SVC)
        if linearity=='nonlinear':
            d1_X = ApproxNonLinear(transX=transX, featX=featX, rho=rho_best, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
            d1_X.fit(X = self.X, 
                     y = Delta_sign,
                     sample_weight=Delta_abs/Delta_abs.mean())
        if linearity=='linear':
            d1_X = clone(self.PipeLinear)
            d1_X.set_params(svm__rho=rho_best, svm__learning_rate=learning_rate)
            d1_X.fit(X = self.X,
                     y = Delta_sign,
                     svm__sample_weight=Delta_abs/Delta_abs.mean())
        if linearity=='tree':
            d1_X = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho_best)
            d1_X.fit(X = self.X,
                        y = Delta_sign,
                        sample_weight=Delta_abs/Delta_abs.mean())
            return d1_X
        
        if self.verbose:
            # save coefficients and value
            # self.d1_X_coef = d1_X.predict(np.concatenate((np.zeros((1,self.X.shape[1])),np.eye(self.X.shape[1]))))
            self.d1_X_coef = np.insert(d1_X.named_steps['svm'].W, 0, d1_X.named_steps['svm'].b)
            # save final value
            d1_X_est = np.sign(d1_X.predict(self.X))
            h0_test = np.zeros_like(d1_X_est)
            if np.sum(d1_X_est==-1)>0:
                h0_test[d1_X_est==-1] = self.predict_h0_a0(data_WX[d1_X_est==-1,:])
            if np.sum(d1_X_est==1)>0:
                h0_test[d1_X_est==1]  = self.predict_h0_a1(data_WX[d1_X_est==1,:])
            self.d1_X_value = np.mean(h0_test)

        def d1_X_predictor(X):
            return np.sign(d1_X.predict(X))
        return d1_X_predictor

    def fit_d1_XZ_cv(self, gamma_f, n_gamma_hs, n_alphas=20, alpha_scales='auto', n_components=50, cv_n=5, cv_r=5, linearity = 'nonlinear', rhos = [0.005, 0.01, 0.05, 0.1, 0.5], learning_rate=None):
        if learning_rate is None:
            learning_rate=self.learning_rate
        # if (not 'gamma_f_h0_a0' in self.__dict__) or (not 'gamma_h_h0_a0' in self.__dict__) or (not 'alpha_scale_h0_a0' in self.__dict__) or (not 'predict_h0_a0' in self.__dict__) or (cv_n != self.cv_h0_a0):
        #     self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        # if (not 'gamma_f_h0_a1' in self.__dict__) or (not 'gamma_h_h0_a1' in self.__dict__) or (not 'alpha_scale_h0_a1' in self.__dict__) or (not 'predict_h0_a1' in self.__dict__) or (cv_n != self.cv_h0_a1):
        #     self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)

        data_WX = np.concatenate((self.W,self.X),axis=1)
        Delta_est = self.predict_h0_a1(data_WX) - self.predict_h0_a0(data_WX)
        Delta_sign = np.sign(Delta_est)
        Delta_abs  = np.abs(Delta_est)
        data_XZ = np.concatenate((self.X, self.Z), axis=1)
        if linearity=='nonlinear':
            transX = Scaler()
            XZ = transX.fit_transform(data_XZ)
            gamma_SVC = FastOptBW(X = XZ, y = Delta_sign, sample_weight=Delta_abs)
            featX = Nystroem(kernel='rbf', gamma=gamma_SVC, random_state=1, n_components=self.n_components).fit(XZ)

        if (type(rhos) is not float) and (len(rhos)>1):
            values = []
            for it, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(data_XZ,Delta_sign)):
                values.append([])
                predict_h0_a0_train = self.fit_h0_a0(n_components=n_components, gamma_f=self.gamma_f_h0_a0, gamma_h=self.gamma_h_h0_a0, alpha_scale=self.alpha_scale_h0_a0, index=train_index)
                predict_h0_a1_train = self.fit_h0_a1(n_components=n_components, gamma_f=self.gamma_f_h0_a1, gamma_h=self.gamma_h_h0_a1, alpha_scale=self.alpha_scale_h0_a1, index=train_index)
                Delta_est_train = predict_h0_a1_train(data_WX[train_index,:]) - predict_h0_a0_train(data_WX[train_index,:])
                Delta_sign_train = np.sign(Delta_est_train)
                Delta_abs_train  = np.abs(Delta_est_train)
                for rho in rhos:
                    #d1_XZ = SVC(kernel='rbf', C=rho, gamma=gamma_SVC)
                    if linearity=='nonlinear':
                        d1_XZ = ApproxNonLinear(transX=transX, featX=featX, rho=rho, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
                        d1_XZ.fit(X = data_XZ[train_index,:], 
                                    y = Delta_sign_train,
                                    sample_weight=Delta_abs_train/Delta_abs_train.mean())
                    if linearity=='linear':
                        d1_XZ = clone(self.PipeLinear)
                        d1_XZ.set_params(svm__rho=rho, svm__learning_rate=learning_rate)
                        d1_XZ.fit(X = data_XZ[train_index,:], 
                                y = Delta_sign_train,
                                svm__sample_weight=Delta_abs_train/Delta_abs_train.mean())
                    if linearity=='tree':
                        d1_XZ = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho)
                        d1_XZ.fit(X = data_XZ[train_index,:],
                                y = Delta_sign_train,
                                sample_weight=Delta_abs_train/Delta_abs_train.mean())
                    d1_XZ_est = np.sign(d1_XZ.predict(data_XZ[test_index,:]))
                    h0_test = np.zeros_like(d1_XZ_est)
                    if np.sum(d1_XZ_est==-1)>0:
                        #h0_test[d1_XZ_est==-1] = predict_h0_a0_train(data_WX[test_index,:][d1_XZ_est==-1,:]) 
                        h0_test[d1_XZ_est==-1] = self.predict_h0_a0(data_WX[test_index,:][d1_XZ_est==-1,:]) 
                    if np.sum(d1_XZ_est==1)>0:
                        #h0_test[d1_XZ_est==1]  = predict_h0_a1_train(data_WX[test_index,:][d1_XZ_est==1,:])  
                        h0_test[d1_XZ_est==1]  = self.predict_h0_a1(data_WX[test_index,:][d1_XZ_est==1,:])  
                    values[it].append(np.mean(h0_test))

            avg_values = np.mean(np.array(values), axis=0)
            rho_best = rhos[np.argmax(avg_values)]
            
            # record best value of d1_XZ
            self.cv_d1_XZ = np.max(avg_values)

            if self.verbose:
                print("d1_XZ")
                print("rho: ",rhos)
                print("value: ",avg_values)
                print("rho_best:", rho_best)
        else:
            rho_best = rhos

        #d1_XZ = SVC(kernel='rbf', C=rho_best, gamma=gamma_SVC)
        if linearity=='nonlinear':
            d1_XZ = ApproxNonLinear(transX=transX, featX=featX, rho=rho_best, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
            d1_XZ.fit(X = data_XZ, 
                     y = Delta_sign,
                     sample_weight=Delta_abs/Delta_abs.mean())
        if linearity=='linear':
            d1_XZ = clone(self.PipeLinear)
            d1_XZ.set_params(svm__rho=rho_best, svm__learning_rate=learning_rate)
            d1_XZ.fit(X = data_XZ, 
                     y = Delta_sign,
                     svm__sample_weight=Delta_abs/Delta_abs.mean())
        if linearity=='tree':
            d1_XZ = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho_best)
            d1_XZ.fit(X = data_XZ,
                      y = Delta_sign,
                      sample_weight=Delta_abs/Delta_abs.mean())
            return d1_XZ

        if self.verbose:
            # save coefficients and value
            # self.d1_XZ_coef = d1_XZ.predict(np.concatenate((np.zeros((1,data_XZ.shape[1])),np.eye(data_XZ.shape[1]))))
            self.d1_XZ_coef = np.insert(d1_XZ.named_steps['svm'].W, 0, d1_XZ.named_steps['svm'].b)
            # save final value
            d1_XZ_est = np.sign(d1_XZ.predict(data_XZ))
            h0_test = np.zeros_like(d1_XZ_est)
            if np.sum(d1_XZ_est==-1)>0:
                h0_test[d1_XZ_est==-1] = self.predict_h0_a0(data_WX[d1_XZ_est==-1,:])
            if np.sum(d1_XZ_est==1)>0:
                h0_test[d1_XZ_est==1]  = self.predict_h0_a1(data_WX[d1_XZ_est==1,:])
            self.d1_XZ_value = np.mean(h0_test)

        def d1_XZ_predictor(X):
            return np.sign(d1_XZ.predict(X))
        return d1_XZ_predictor

    def fit_d2_XW_cv(self, gamma_f, n_gamma_hs, n_alphas=20, alpha_scales='auto', n_components=50, cv_n=5, cv_r=5, linearity = 'nonlinear', rhos = [0.005, 0.01, 0.05, 0.1, 0.5], learning_rate=None):
        if learning_rate is None:
            learning_rate=self.learning_rate
        # if (not 'q0_est' in self.__dict__) or (self.cv_q0_a0 != cv_n) or (self.cv_q0_a1 != cv_n):
        #     self.predict_q0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        self.predict_q0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)

        ayq0_sign = np.sign(self.A-0.5)*np.sign(self.Y)*np.sign(self.q0_est)
        yq0_abs  = np.abs(self.Y*self.q0_est)
        data_XW = np.concatenate((self.X, self.W), axis=1)
        if linearity == 'nonlinear':
            transX = Scaler()
            XW = transX.fit_transform(data_XW)
            gamma_SVC = FastOptBW(X = XW, y = ayq0_sign, sample_weight=yq0_abs)
            featX = Nystroem(kernel='rbf', gamma=gamma_SVC, random_state=1, n_components=self.n_components).fit(XW)
        
        if (type(rhos) is not float) and (len(rhos)>1):
            values = []
            for it, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(data_XW,ayq0_sign)):
                values.append([])
                yq0_train = self.Y[train_index]*self.predict_q0(n_components=n_components, index=train_index)
                for rho in rhos:
                    #d2 = SVC(kernel='rbf', C=rho, gamma=gamma_SVC)
                    if linearity == 'nonlinear':
                        d2 = ApproxNonLinear(transX=transX, featX=featX, rho=rho, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
                        d2.fit(X = data_XW[train_index,:], 
                                y = np.sign(self.A[train_index]-0.5)*np.sign(yq0_train),
                                sample_weight=np.abs(yq0_train)/np.mean(np.abs(yq0_train)))
                    if linearity=='linear':
                        d2 = clone(self.PipeLinear)
                        d2.set_params(svm__rho=rho, svm__learning_rate=learning_rate)
                        d2.fit(X = data_XW[train_index,:], 
                            y = np.sign(self.A[train_index]-0.5)*np.sign(yq0_train),
                            svm__sample_weight=np.abs(yq0_train)/np.mean(np.abs(yq0_train)))
                    if linearity=='tree':
                        d2 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho)
                        d2.fit(X = data_XW[train_index,:],
                            y = np.sign(self.A[train_index]-0.5)*np.sign(yq0_train),
                            sample_weight=np.abs(yq0_train)/np.mean(np.abs(yq0_train)))
                    d2_est = np.sign(d2.predict(data_XW[test_index,:]))
                    values[it].append(np.mean(self.Y[test_index]*self.q0_est[test_index]*(d2_est==(np.sign(self.A[test_index]-0.5)))))

            avg_values = np.mean(np.array(values), axis=0)
            rho_best = rhos[np.argmax(avg_values)]

            # record best value of d2_XW
            self.cv_d2_XW = np.max(avg_values)

            if self.verbose:
                print("d2_XW")
                print("rho: ",rhos)
                print("value: ",avg_values)
                print("rho_best:", rho_best)
        else:
            rho_best = rhos

        #d2 = SVC(kernel='rbf', C=rho_best, gamma=gamma_SVC)
        if linearity=='nonlinear':
            d2 = ApproxNonLinear(transX=transX, featX=featX, rho=rho_best, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
            d2.fit(X = data_XW, 
                   y = ayq0_sign,
                   sample_weight=yq0_abs/yq0_abs.mean())
        if linearity=='linear':
            d2 = clone(self.PipeLinear)
            d2.set_params(svm__rho=rho_best, svm__learning_rate=learning_rate)
            d2.fit(X = data_XW, 
                   y = ayq0_sign,
                   svm__sample_weight=yq0_abs/yq0_abs.mean())
        if linearity=='tree':
            d2 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho_best)
            d2.fit(X = data_XW, 
                   y = ayq0_sign,
                   sample_weight=yq0_abs/yq0_abs.mean())
            return d2

        if self.verbose:
            # save coefficients and value
            # self.d2_coef = d2.predict(np.concatenate((np.zeros((1,data_XW.shape[1])),np.eye(data_XW.shape[1]))))
            self.d2_coef = np.insert(d2.named_steps['svm'].W, 0, d2.named_steps['svm'].b)
            # save final value
            d2_est = np.sign(d2.predict(data_XW))
            self.d2_value = np.mean(self.Y*self.q0_est*(d2_est==(np.sign(self.A-0.5))))

        def d2_predictor(X):
            return np.sign(d2.predict(X))
        return d2_predictor
        
    def fit_d2_X_cv(self, gamma_f, n_gamma_hs, n_alphas=20, alpha_scales='auto', n_components=50, cv_n=5, cv_r=5, linearity = 'nonlinear', rhos = [0.005, 0.01, 0.05, 0.1, 0.5], learning_rate=None):
        if learning_rate is None:
            learning_rate=self.learning_rate
        # if (not 'q0_est' in self.__dict__) or (self.cv_q0_a0 != cv_n) or (self.cv_q0_a1 != cv_n):
        #     self.predict_q0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)
        self.predict_q0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=alpha_scales, n_components=n_components, cv=cv_n)

        ayq0_sign = np.sign(self.A-0.5)*np.sign(self.Y)*np.sign(self.q0_est)
        yq0_abs  = np.abs(self.Y*self.q0_est)
        if linearity=='nonlinear':
            transX = Scaler()
            X = transX.fit_transform(self.X)
            gamma_SVC = FastOptBW(X = X, y = ayq0_sign, sample_weight=yq0_abs)
            featX = Nystroem(kernel='rbf', gamma=gamma_SVC, random_state=1, n_components=self.n_components).fit(X)
        
        if (type(rhos) is not float) and (len(rhos)>1):
            values = []
            for it, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X,ayq0_sign)):
                values.append([])
                yq0_train = self.Y[train_index]*self.predict_q0(n_components=n_components, index=train_index)
                for rho in rhos:
                    #d3 = SVC(kernel='rbf', C=rho, gamma=gamma_SVC)
                    if linearity == 'nonlinear':
                        d3 = ApproxNonLinear(transX=transX, featX=featX, rho=rho, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
                        d3.fit(X = self.X[train_index,:], 
                                y = np.sign(self.A[train_index]-0.5)*np.sign(yq0_train),
                                sample_weight=np.abs(yq0_train)/np.mean(np.abs(yq0_train)))
                    if linearity == 'linear':
                        d3 = clone(self.PipeLinear)
                        d3.set_params(svm__rho=rho, svm__learning_rate=learning_rate)
                        d3.fit(X = self.X[train_index,:], 
                            y = np.sign(self.A[train_index]-0.5)*np.sign(yq0_train),
                            svm__sample_weight=np.abs(yq0_train)/np.mean(np.abs(yq0_train)))
                    if linearity == 'tree':
                        d3 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho)
                        d3.fit(X = self.X[train_index,:], 
                                y = np.sign(self.A[train_index]-0.5)*np.sign(yq0_train),
                                sample_weight=np.abs(yq0_train)/np.mean(np.abs(yq0_train)))

                    d3_est = np.sign(d3.predict(self.X[test_index,:]))
                    values[it].append(np.mean(self.Y[test_index]*self.q0_est[test_index]*(d3_est==(np.sign(self.A[test_index]-0.5)))))

            avg_values = np.mean(np.array(values), axis=0)
            rho_best = rhos[np.argmax(avg_values)]

            if self.verbose:
                print("d3_X")
                print("rho: ",rhos)
                print("value: ",avg_values)
                print("rho_best:", rho_best)
        else:
            rho_best = rhos

        #d3 = SVC(kernel='rbf', C=rho_best, gamma=gamma_SVC)
        if linearity == 'nonlinear':
            d3 = ApproxNonLinear(transX=transX, featX=featX, rho=rho_best, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
            d3.fit(X = self.X,
                   y = ayq0_sign,
                   sample_weight=yq0_abs/yq0_abs.mean())
        if linearity == 'linear':
            d3 = clone(self.PipeLinear)
            d3.set_params(svm__rho=rho_best, svm__learning_rate=learning_rate)
            d3.fit(X = self.X, 
                   y = ayq0_sign,
                   svm__sample_weight=yq0_abs/yq0_abs.mean())
        if linearity == 'tree':
            d3 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho_best)
            d3.fit(X = self.X,
                   y = ayq0_sign,
                   sample_weight=yq0_abs/yq0_abs.mean())
            return d3

        if self.verbose:
            # save coefficients and value
            # self.d3_coef = d3.predict(np.concatenate((np.zeros((1,self.X.shape[1])),np.eye(self.X.shape[1]))))
            self.d3_coef = np.insert(d3.named_steps['svm'].W, 0, d3.named_steps['svm'].b)
            # save final value
            d3_est = np.sign(d3.predict(self.X))
            self.d3_value = np.mean(self.Y*self.q0_est*(d3_est==(np.sign(self.A-0.5))))

        def d3_predictor(X):
            return np.sign(d3.predict(X))
        return d3_predictor

    
    def C0C1(self, paras, n_components=20, train_index='all', test_index='all'):
        if _check_all(train_index):
            train_index = range(self.A.shape[0])
        if _check_all(test_index):
            test_index = range(self.A.shape[0])

        (gamma_f_h0_a0, gamma_h_h0_a0, alpha_scale_h0_a0,\
        gamma_f_h0_a1, gamma_h_h0_a1, alpha_scale_h0_a1,\
        gamma_f_q0_a0, gamma_h_q0_a0, alpha_scale_q0_a0,\
        gamma_f_q0_a1, gamma_h_q0_a1, alpha_scale_q0_a1) = paras

        pred_h0_a0\
             =self.fit_h0_a0(gamma_f=gamma_f_h0_a0, gamma_h=gamma_h_h0_a0, alpha_scale=alpha_scale_h0_a0, n_components=n_components, index=train_index)
        pred_h0_a1\
             =self.fit_h0_a1(gamma_f=gamma_f_h0_a1, gamma_h=gamma_h_h0_a1, alpha_scale=alpha_scale_h0_a1, n_components=n_components, index=train_index)
        pred_q0_a0\
             =self.fit_q0_a0(gamma_f=gamma_f_q0_a0, gamma_h=gamma_h_q0_a0, alpha_scale=alpha_scale_q0_a0, n_components=n_components, index=train_index)
        pred_q0_a1\
             =self.fit_q0_a1(gamma_f=gamma_f_q0_a1, gamma_h=gamma_h_q0_a1, alpha_scale=alpha_scale_q0_a1, n_components=n_components, index=train_index)
        h0_a0_est = pred_h0_a0(np.concatenate((self.W,self.X),axis=1)[test_index,:])
        h0_a1_est = pred_h0_a1(np.concatenate((self.W,self.X),axis=1)[test_index,:])
        q0_a0_est = pred_q0_a0(np.concatenate((self.Z,self.X),axis=1)[test_index,:])
        q0_a1_est = pred_q0_a1(np.concatenate((self.Z,self.X),axis=1)[test_index,:])
        C0 = h0_a0_est + (1-self.A[test_index]) * q0_a0_est*(self.Y[test_index]-h0_a0_est)
        C1 = h0_a1_est +     self.A[test_index] * q0_a1_est*(self.Y[test_index]-h0_a1_est)
        return C0, C1

    def C0C1_cv(self, gamma_f, n_gamma_hs, n_alphas=20, h_alpha_scales='auto', q_alpha_scales='auto', n_components=20, cv=5, train_index='all', test_index='all'):
        if _check_all(train_index):
            train_index = range(self.A.shape[0])
        if _check_all(test_index):
            test_index = range(self.A.shape[0])

        gamma_f_h0_a0, gamma_h_h0_a0, alpha_scale_h0_a0, pred_h0_a0\
            =self.fit_h0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=h_alpha_scales, n_components=n_components, cv=cv, index=train_index)
        gamma_f_h0_a1, gamma_h_h0_a1, alpha_scale_h0_a1, pred_h0_a1\
            =self.fit_h0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=h_alpha_scales, n_components=n_components, cv=cv, index=train_index)
        gamma_f_q0_a0, gamma_h_q0_a0, alpha_scale_q0_a0, pred_q0_a0\
            =self.fit_q0_a0_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=q_alpha_scales, n_components=n_components, cv=cv, index=train_index)
        gamma_f_q0_a1, gamma_h_q0_a1, alpha_scale_q0_a1, pred_q0_a1\
            =self.fit_q0_a1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, alpha_scales=q_alpha_scales, n_components=n_components, cv=cv, index=train_index)
        h0_a0_est = pred_h0_a0(np.concatenate((self.W,self.X),axis=1)[test_index,:])
        h0_a1_est = pred_h0_a1(np.concatenate((self.W,self.X),axis=1)[test_index,:])
        q0_a0_est = pred_q0_a0(np.concatenate((self.Z,self.X),axis=1)[test_index,:])
        q0_a1_est = pred_q0_a1(np.concatenate((self.Z,self.X),axis=1)[test_index,:])
        C0 = h0_a0_est + (1-self.A[test_index]) * q0_a0_est*(self.Y[test_index]-h0_a0_est)
        C1 = h0_a1_est +     self.A[test_index] * q0_a1_est*(self.Y[test_index]-h0_a1_est)
        return C0, C1,\
            (gamma_f_h0_a0, gamma_h_h0_a0, alpha_scale_h0_a0,\
                gamma_f_h0_a1, gamma_h_h0_a1, alpha_scale_h0_a1,\
                    gamma_f_q0_a0, gamma_h_q0_a0, alpha_scale_q0_a0,\
                        gamma_f_q0_a1, gamma_h_q0_a1, alpha_scale_q0_a1)

    def eif_CI(self, gamma_f, n_gamma_hs, n_alphas=20, h_alpha_scales='auto', q_alpha_scales='auto',
                    n_components=50, cv_n=5, cv_r=3):
        C0_all, C1_all, paras = self.C0C1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, h_alpha_scales=h_alpha_scales, q_alpha_scales=q_alpha_scales,
                                    n_components=n_components, cv=cv_n, 
                                    train_index='all',test_index='all')
        C1_C0_sign_cv = np.sign(C1_all-C0_all)

        Values = []
        for it, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X,C1_C0_sign_cv)):
            C0, C1 = self.C0C1(paras = paras, n_components=n_components,
                               train_index=train_index, test_index=test_index)
            Values.append(C0*(self.d_X_opt[test_index]<0) + C1*(self.d_X_opt[test_index]>0))
        Values = np.concatenate(Values)

        CI = stats.norm.interval(0.95, loc=np.mean(Values), scale=np.std(Values)/np.sqrt(len(Values)))
        
        return CI

    def fit_d_DR_cv(self, gamma_f, n_gamma_hs, n_alphas=20, h_alpha_scales='auto', q_alpha_scales='auto',
                    n_components=50, cv_n=5, cv_r=3, linearity = 'nonlinear', rhos = [0.005, 0.01, 0.05, 0.1, 0.5], learning_rate=None):
        if learning_rate is None:
            learning_rate=self.learning_rate
        C0_all, C1_all, paras = self.C0C1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, h_alpha_scales=h_alpha_scales, q_alpha_scales=q_alpha_scales,
                                    n_components=n_components, cv=cv_n, 
                                    train_index='all',test_index='all')
        C1_C0_sign_cv = np.sign(C1_all-C0_all)
        if linearity == 'nonlinear':
            transX = Scaler()
            X = transX.fit_transform(self.X)
            gamma_SVC = FastOptBW(X = X,
                                       y = np.sign(C1_all-C0_all),
                                       sample_weight=np.abs(C1_all - C0_all))
            featX = Nystroem(kernel='rbf', gamma=gamma_SVC, random_state=1, n_components=self.n_components).fit(X)

        if (type(rhos) is not float) and (len(rhos)>1):
            values = []
            for it1, (TRAIN_INDEX, TEST_INDEX) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X, C1_C0_sign_cv)):
                values.append([])
                d_DR_est = np.zeros((len(TEST_INDEX), len(rhos)))
                #for train_index, test_index in StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X[TRAIN_INDEX,:],C1_C0_sign_cv[TRAIN_INDEX]):
                for test_index, train_index in StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X[TRAIN_INDEX,:],C1_C0_sign_cv[TRAIN_INDEX]):
                    C0, C1 = C0_all[TRAIN_INDEX[test_index]], C1_all[TRAIN_INDEX[test_index]]
                    for it2, rho in enumerate(rhos):
                        if linearity == 'nonlinear':
                            d_DR = ApproxNonLinear(transX=transX, featX=featX, rho=rho, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
                            d_DR.fit(X = self.X[TRAIN_INDEX,:][test_index,:],
                                        y = np.sign(C1-C0),
                                        sample_weight = np.abs(C1-C0)/np.mean(np.abs(C1-C0)))
                        if linearity == 'linear':
                            d_DR = clone(self.PipeLinear)
                            d_DR.set_params(svm__rho=rho, svm__learning_rate=learning_rate)
                            d_DR.fit(X = self.X[TRAIN_INDEX,:][test_index,:],
                                    y = np.sign(C1-C0),
                                    svm__sample_weight = np.abs(C1-C0)/np.mean(np.abs(C1-C0)))

                        d_DR_est[:,it2] = d_DR_est[:,it2] + d_DR.predict(self.X[TEST_INDEX,:])
                for k in range(len(rhos)):
                    values[it1].append((np.sum(C0_all[TEST_INDEX][d_DR_est[:,k]<0])+np.sum(C1_all[TEST_INDEX][d_DR_est[:,k]>=0]))/len(TEST_INDEX))
            
            avg_values = np.mean(np.array(values), axis=0)
            rho_best = rhos[np.argmax(avg_values)]
            if self.verbose:
                print("d_DR")
                print("rho: ",rhos)
                print("value: ",avg_values)
                print("rho_best:", rho_best)
        else:
            rho_best = rhos

        predictors = []
        d_DR_coef = np.zeros((cv_r, self.X.shape[1]+1))
        #for it, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X,C1_C0_sign_cv)):
        for it, (test_index, train_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X,C1_C0_sign_cv)):
            C0, C1 = C0_all[test_index], C1_all[test_index]
            if linearity == 'nonlinear':
                d_DR = ApproxNonLinear(transX=transX, featX=featX, rho=rho_best, n_epoch=self.n_epoch, batch_size = self.batch_size, learning_rate=learning_rate, opt=self.opt)
                d_DR.fit(X = self.X[test_index,:],
                         y = np.sign(C1-C0),
                         sample_weight = np.abs(C1-C0)/np.mean(np.abs(C1-C0)))
            if linearity == 'linear':
                d_DR = clone(self.PipeLinear)
                d_DR.set_params(svm__rho=rho_best, svm__learning_rate=learning_rate)
                d_DR.fit(X = self.X[test_index,:],
                        y = np.sign(C1-C0),
                        svm__sample_weight = np.abs(C1-C0)/np.mean(np.abs(C1-C0)))
                d_DR_coef[it,:] = np.insert(d_DR.named_steps['svm'].W, 0, d_DR.named_steps['svm'].b)
            predictors.append(d_DR.predict)
        
        def d_DR_predict(X):
            d_DR_est = np.zeros(X.shape[0])
            for predictor in predictors:
                d_DR_est = d_DR_est + predictor(X)
            return d_DR_est/len(predictors)
        
        def d_DR_predictor(X):
            return np.sign(d_DR_predict(X))

        if self.verbose:
            # save coefficients and value
            #self.d_DR_coef = d_DR_predict(np.concatenate((np.zeros((1,self.X.shape[1])),np.eye(self.X.shape[1]))))
            self.d_DR_coef = np.mean(d_DR_coef, axis=0)
            # save final value
            d_DR_est = d_DR_predictor(self.X)
            self.d_DR_value = (np.sum(C0_all[d_DR_est<0])+np.sum(C1_all[d_DR_est>=0]))/self.X.shape[0]

        return d_DR_predictor

    def fit_d_DR_tree(self, gamma_f, n_gamma_hs, n_alphas=20, h_alpha_scales='auto', q_alpha_scales='auto',
                    n_components=50, cv_n=5, cv_r=5, linearity = 'tree', rhos = [0.005, 0.01, 0.05, 0.1, 0.5], learning_rate=None):
        if learning_rate is None:
            learning_rate=self.learning_rate
        C0_all, C1_all, paras = self.C0C1_cv(gamma_f=gamma_f, n_gamma_hs=n_gamma_hs, n_alphas=n_alphas, h_alpha_scales=h_alpha_scales, q_alpha_scales=q_alpha_scales,
                                    n_components=n_components, cv=cv_n, 
                                    train_index='all',test_index='all')
        C1_C0_sign_cv = np.sign(C1_all-C0_all)

        values = []
        for it, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=cv_r, random_state=None, shuffle=False).split(self.X,C1_C0_sign_cv)):
            values.append([])
            C0, C1 = C0_all[train_index], C1_all[train_index]
            for rho in rhos:
                d_DR = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho)
                d_DR.fit(X = self.X[train_index,:],
                            y = np.sign(C1-C0),
                            sample_weight = np.abs(C1-C0)/np.mean(np.abs(C1-C0)))
                d_DR_est = d_DR.predict(self.X[test_index,:])
                values[it].append((np.sum(C0_all[test_index][d_DR_est<0])+np.sum(C1_all[test_index][d_DR_est>=0]))/len(test_index))

        avg_values = np.mean(np.array(values), axis=0)
        rho_best = rhos[np.argmax(avg_values)]

        if self.verbose:
            print("d_DR")
            print("rho: ",rhos)
            print("value: ",avg_values)
            print("rho_best:", rho_best)
        
        d_DR = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=rho_best)
        d_DR.fit(X = self.X,
                 y = np.sign(C1_all-C0_all),
                 sample_weight = np.abs(C1_all-C0_all)/np.mean(np.abs(C1_all-C0_all)))

        return d_DR
