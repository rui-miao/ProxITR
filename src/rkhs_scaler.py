# Copyright (c) Rui Miao 2021-2022
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler as Scaler
#from sklearn.preprocessing import StandardScaler as Scaler


def _check_auto(param):
    return (isinstance(param, str) and (param == 'auto'))


class _BaseRKHSIV:

    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n):
        '''
        delta -> Critical radius
        '''
        delta_scale = 5 if _check_auto(self.delta_scale) else self.delta_scale
        delta_exp = .4 if _check_auto(self.delta_exp) else self.delta_exp
        return delta_scale / (n**(delta_exp))

    def _get_alpha_scale(self):
        return 60 if _check_auto(self.alpha_scale) else self.alpha_scale

    def _get_alpha_scales(self):
        return ([c for c in np.geomspace(0.1, 1e5, self.n_alphas)]
                if _check_auto(self.alpha_scales) else self.alpha_scales)

    def _get_alpha(self, delta, alpha_scale):
        return alpha_scale * (delta**4)

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def _get_gamma_f(self, condition):
        if _check_auto(self.gamma_f):
            params = {"squared": True}
            K_condition_euclidean = sklearn.metrics.pairwise_distances(X = condition, metric='euclidean', n_jobs=-1, **params)
            # gamma_f = 1./(condition.shape[1] * np.median(K_condition_euclidean[np.tril_indices(condition.shape[0],-1)]))
            gamma_f = 1./(np.median(K_condition_euclidean[np.tril_indices(condition.shape[0],-1)]))
            return gamma_f
        else:
            return self.gamma_f

    def _get_kernel_f(self, X, Y=None, gamma_f=0.1):
        params = {"gamma": gamma_f}
        return pairwise_kernels(X, Y, metric='rbf', filter_params=True, **params)

    def _get_kernel_h(self, X, Y=None, gamma_h=0.01):
        params = {"gamma": gamma_h}
        return pairwise_kernels(X, Y, metric='rbf', filter_params=True, **params)


class RKHSIV(BaseEstimator, _BaseRKHSIV):

    def __init__(self, gamma_h=0.1, gamma_f='auto', 
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto'):
        """
        Parameters:
            gamma_h : the gamma parameter for the rbf kernel of h
            gamma_f : the gamma parameter for the rbf kernel of f
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
        """
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h 
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5

    def fit(self, X, y, condition):
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and get gamma_f -> Kf -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition=condition)
        self.gamma_f = gamma_f
        Kf = self._get_kernel_f(condition, gamma_f=self.gamma_f)
        RootKf = scipy.linalg.sqrtm(Kf).astype(float)

        # Standardize X and get Kh
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.X = X.copy()
        Kh = self._get_kernel_h(X, gamma_h=self.gamma_h)

        # delta & alpha
        n = X.shape[0]  # number of samples
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        # M
        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf

        #self.a = np.linalg.pinv(Kh @ M @ Kh + alpha * Kh) @ Kh @ M @ y
        self.a = np.linalg.lstsq(Kh @ M @ Kh + alpha * Kh, Kh @ M @ y, rcond=None)[0]
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        return self._get_kernel_h(X, Y=self.X, gamma_h=self.gamma_h) @ self.a

    def score(self, X, y, M):
        n = X.shape[0]
        #delta = self._get_delta(n)
        #Kf = self._get_kernel_f(Z, gamma_f=self.gamma_f)
        #RootKf = scipy.linalg.sqrtm(Kf).astype(float)
        #M = RootKf @ np.linalg.inv(
        #    Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        y_pred = self.predict(X)
        return ((y - y_pred).T @ M @ (y - y_pred)).reshape(-1)[0] / n**2

class RKHSIVCV(RKHSIV):

    def __init__(self, gamma_f='auto', gamma_hs='auto', n_gamma_hs=20,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            gamma_f : the gamma parameter for the kernel of f
            gamma_hs : the list of gamma parameters for kernel of h
            n_gamma_hs : how many gamma_hs to try
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how many alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale, gamma_h
        """

        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs=n_gamma_hs
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv

    def _get_gamma_hs(self,X):
        if _check_auto(self.gamma_hs):
            params = {"squared": True}
            K_X_euclidean = sklearn.metrics.pairwise_distances(X = X, metric='euclidean', **params)
            #return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)/X.shape[1]
            return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)
        else:
            return self.gamma_hs

    def fit(self, X, y, condition):
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition = condition)
        Kf = self._get_kernel_f(condition, gamma_f=gamma_f)
        RootKf = scipy.linalg.sqrtm(Kf).astype(float)

        # Standardize X and get gamma_hs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.X = X.copy()
        gamma_hs = self._get_gamma_hs(X)
        #Khs = [self._get_kernel_h(X, gamma_h = gammah) for gammah in gamma_hs]

        # delta & alpha
        n = X.shape[0]
        n_train = n * (self.cv - 1) / self.cv
        delta_train = self._get_delta(n_train)
        n_test = n / self.cv
        delta_test = self._get_delta(n_test)
        alpha_scales = self._get_alpha_scales()

        # get best (alpha, gamma_h) START
        scores = []
        for it1, (train, test) in enumerate(KFold(n_splits=self.cv).split(X)):
            # Standardize X_train
            transX = Scaler()
            X_train = transX.fit_transform(X[train])
            X_test = transX.transform(X[test])
            # Standardize condition_train and get Kf_train, RootKf_train, M_train
            condition_train = Scaler().fit_transform(condition[train])
            Kf_train = self._get_kernel_f(X=condition_train, gamma_f=self._get_gamma_f(condition=condition_train))
            RootKf_train = scipy.linalg.sqrtm(Kf_train).astype(float)
            M_train = RootKf_train @ np.linalg.inv(
                Kf_train / (2 * n_train * (delta_train**2)) + np.eye(len(train)) / 2) @ RootKf_train
            # Use M_test based on precomputed RootKf to make sure evaluations are the same
            M_test = RootKf[np.ix_(test, test)] @ np.linalg.inv(
                Kf[np.ix_(test, test)] / (2 * n_test * (delta_test**2)) + np.eye(len(test)) / 2) @ RootKf[np.ix_(test, test)]
            scores.append([])
            for it2, gamma_h in enumerate(gamma_hs):
                Kh_train = self._get_kernel_h(X=X_train, gamma_h=gamma_h)
                KMK_train = Kh_train @ M_train @ Kh_train
                B_train = Kh_train @ M_train @ y[train]
                scores[it1].append([])
                for alpha_scale in alpha_scales:
                    alpha = self._get_alpha(delta_train, alpha_scale)
                    #a = np.linalg.pinv(KMK_train + alpha * Kh_train) @ B_train
                    a = np.linalg.lstsq(KMK_train + alpha * Kh_train, B_train, rcond=None)[0]
                    res = y[test] - self._get_kernel_h(X=X_test, Y=X_train, gamma_h=gamma_h) @ a
                    scores[it1][it2].append((res.T @ M_test @ res).reshape(-1)[0] / (res.shape[0]**2))

        avg_scores = np.mean(np.array(scores), axis=0)
        best_ind = np.unravel_index(np.argmin(avg_scores), avg_scores.shape)
        self.gamma_h = gamma_hs[best_ind[0]]
        self.best_alpha_scale = alpha_scales[best_ind[1]]
        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)
        # M
        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        # Kh
        Kh = self._get_kernel_h(X, gamma_h=self.gamma_h)

        # self.a = np.linalg.pinv(
        #     Kh @ M @ Kh + self.best_alpha * Kh) @ Kh @ M @ y
        self.a = np.linalg.lstsq(
            Kh @ M @ Kh + self.best_alpha * Kh, Kh @ M @ y, rcond=None)[0]

        return self

class ApproxRKHSIV(_BaseRKHSIV):

    def __init__(self, kernel_approx='nystrom', n_components=25,
                 gamma_f='auto', gamma_h=0.1,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto'):
        """
        Parameters:
            kernel_approx : what approximator to use; either 'nystrom' or 'rbfsampler' (for kitchen sinks)
            n_components : how many approximation components to use
            # kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma_h : the gamma parameter for the kernel of h
            gamma_f : the gamma parameter for the kernel of f
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h 
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5

    def _get_new_approx_instance(self, gamma):
        if self.kernel_approx == 'rbfsampler':
            return RBFSampler(gamma=gamma, n_components=self.n_components, random_state=1)
        elif self.kernel_approx == 'nystrom':
            return Nystroem(kernel='rbf', gamma=gamma, random_state=1, n_components=self.n_components)
        else:
            raise AttributeError("Invalid kernel approximator")

    def fit(self, X, y, condition):
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition=condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=self.gamma_f)
        RootKf = self.featCond.fit_transform(condition)

        # Standardize X and get gamma_hs -> RootKhs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)

        # delta & alpha
        n = X.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ y
        # self.a = np.linalg.pinv(W) @ B
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        return self.featX.transform(X) @ self.a


class ApproxRKHSIVCV(ApproxRKHSIV):

    def __init__(self, kernel_approx='nystrom', n_components=25,
                 gamma_f='auto', gamma_hs = 'auto', n_gamma_hs=10,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            kernel_approx : what approximator to use; either 'nystrom' or 'rbfsampler' (for kitchen sinks)
            n_components : how many nystrom components to use
            gamma_f : the gamma parameter for the kernel of f
            gamma_hs : the list of gamma parameters for kernel of h
            n_gamma_hs : how many gamma_hs to try
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how mny alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components

        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs=n_gamma_hs

        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv

    def _get_gamma_hs(self,X):
        if _check_auto(self.gamma_hs):
            params = {"squared": True}
            K_X_euclidean = sklearn.metrics.pairwise_distances(X = X, metric='euclidean', **params)
            #return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)/X.shape[1]
            return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)
        else:
            return self.gamma_hs

    def fit(self, X, y, condition):
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition = condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=gamma_f)
        RootKf = self.featCond.fit_transform(condition)

        # Standardize X and get gamma_hs -> RootKhs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        gamma_hs = self._get_gamma_hs(X)
        RootKhs = [self._get_new_approx_instance(gamma=gammah).fit_transform(X) for gammah in gamma_hs]

        # delta & alpha
        n = X.shape[0]
        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)

        scores = []
        for it1, (train, test) in enumerate(KFold(n_splits=self.cv).split(X)):
            RootKf_train, RootKf_test = RootKf[train], RootKf[test]
            Q_train = np.linalg.pinv(
                RootKf_train.T @ RootKf_train / (2 * n_train * (delta_train**2)) + np.eye(self.n_components) / 2)
            Q_test = np.linalg.pinv(
                RootKf_test.T @ RootKf_test / (2 * n_test * (delta_test**2)) + np.eye(self.n_components) / 2)
            scores.append([])
            for it2, RootKh in enumerate(RootKhs):
                RootKh_train, RootKh_test = RootKh[train], RootKh[test]
                A_train = RootKh_train.T @ RootKf_train
                AQA_train = A_train @ Q_train @ A_train.T
                B_train = A_train @ Q_train @ RootKf_train.T @ y[train]
                scores[it1].append([])
                for alpha_scale in alpha_scales:
                    alpha = self._get_alpha(delta_train, alpha_scale)
                    # a = np.linalg.pinv(AQA_train + alpha *
                    #                    np.eye(self.n_components)) @ B_train
                    a = np.linalg.lstsq(AQA_train + alpha *
                                       np.eye(self.n_components), B_train, rcond=None)[0]
                    res = RootKf_test.T @ (y[test] - RootKh_test @ a)
                    scores[it1][it2].append((res.T @ Q_test @ res).reshape(-1)[0] / (len(test)**2))

        avg_scores = np.mean(np.array(scores), axis=0)
        best_ind = np.unravel_index(np.argmin(avg_scores), avg_scores.shape)

        self.gamma_h = gamma_hs[best_ind[0]]
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)

        self.best_alpha_scale = alpha_scales[best_ind[1]]
        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + self.best_alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ y
        # self.a = np.linalg.pinv(W) @ B
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self

class RKHSIV_q(_BaseRKHSIV):

    def __init__(self, gamma_h=0.1, gamma_f='auto', 
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto'):
        """
        Parameters:
            gamma_h : the gamma parameter for the rbf kernel of h
            gamma_f : the gamma parameter for the rbf kernel of f
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
        """
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h 
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5

    def fit(self, X, y, condition, index):
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition=condition)
        self.gamma_f = gamma_f
        Kf = self._get_kernel_f(condition, gamma_f=gamma_f)
        RootKf = scipy.linalg.sqrtm(Kf).astype(float)

        # Standardize X and get Kh, Kh0
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.X = X.copy()
        Kh = self._get_kernel_h(X, gamma_h=self.gamma_h)
        Kh0 = np.zeros_like(Kh)
        Kh0[index,:] = Kh[index,:]

        # delta & alpha
        n = X.shape[0]  # number of samples
        #delta = self._get_delta(n)
        delta = self._get_delta(np.sum(index)) # only sum(index) of effective data
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf

        # self.a = np.linalg.pinv(Kh0.T @ M @ Kh0 + alpha * Kh) @ Kh0 @ M @ y
        self.a = np.linalg.lstsq(Kh0.T @ M @ Kh0 + alpha * Kh, Kh0 @ M @ y, rcond=None)[0]
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        return self._get_kernel_h(X, Y=self.X, gamma_h=self.gamma_h) @ self.a

    def score(self, X, y, M, index):
        n = X.shape[0]
        #Kf = self._get_kernel_f(Z, gamma_f=self.gamma_f)
        #RootKf = scipy.linalg.sqrtm(Kf).astype(float)
        #M = RootKf @ np.linalg.inv(
        #    Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        y_pred = np.zeros_like(y)
        y_pred[index] = self.predict(X[index,:])
        return ((y - y_pred).T @ M @ (y - y_pred)).reshape(-1)[0] / n**2

class RKHSIVCV_q(RKHSIV_q):

    def __init__(self, gamma_f='auto', gamma_hs='auto', n_gamma_hs=25,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            gamma_f : the gamma parameter for the kernel of f
            gamma_hs : the list of gamma parameters for kernel of h
            n_gamma_hs : how many gamma_hs to try
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how many alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale, gamma_h
        """
        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs=n_gamma_hs
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv

    def _get_gamma_hs(self,X):
        if _check_auto(self.gamma_hs):
            params = {"squared": True}
            K_X_euclidean = sklearn.metrics.pairwise_distances(X = X, metric='euclidean', **params)
            # return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)/X.shape[1]
            return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)
        else:
            return self.gamma_hs

    def fit(self, X, y, condition, index):
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and et gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition = condition)
        Kf = self._get_kernel_f(condition, gamma_f=gamma_f)
        RootKf = scipy.linalg.sqrtm(Kf).astype(float)

        # Standardize X and get gamma_hs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.X = X.copy()
        gamma_hs = self._get_gamma_hs(X)
        #Khs = []
        #for gammah in gamma_hs:
        #    Kh = self._get_kernel_h(X, gamma_h = gammah)
        #    Kh0 = np.zeros_like(Kh)
        #    Kh0[index,:] = Kh[index,:]
        #    Khs.append((Kh, Kh0))

        # delta & alpha
        n = X.shape[0]
        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv

        # get best (alpha, gamma_h) START
        scores = []
        for it1, (train, test) in enumerate(StratifiedKFold(n_splits=self.cv).split(X, index)):
            # Standardize X_train
            transX = Scaler()
            X_train = transX.fit_transform(X[train])
            X_test = transX.transform(X[test])
            # Standardize condition_train and get Kf_train, RootKf_train, M_train
            condition_train = Scaler().fit_transform(condition[train])
            Kf_train = self._get_kernel_f(X=condition_train, gamma_f=self._get_gamma_f(condition=condition_train))
            RootKf_train = scipy.linalg.sqrtm(Kf_train).astype(float)
            delta_train = self._get_delta(np.sum(index[train]))
            M_train = RootKf_train @ np.linalg.inv(
                Kf_train / (2 * n_train * (delta_train**2)) + np.eye(len(train)) / 2) @ RootKf_train
            # Use M_test based on precomputed RootKf to make sure evaluations are the same
            delta_test = self._get_delta(np.sum(index[test]))
            M_test = RootKf[np.ix_(test, test)] @ np.linalg.inv(
                Kf[np.ix_(test, test)] / (2 * n_test * (delta_test**2)) + np.eye(len(test)) / 2) @ RootKf[np.ix_(test, test)]
            scores.append([])
            for it2, gamma_h in enumerate(gamma_hs):
                Kh_train = self._get_kernel_h(X_train, gamma_h = gamma_h)
                Kh0_train = np.zeros_like(Kh_train)
                Kh0_train[index[train],:] = Kh_train[index[train],:]
                KMK_train = Kh0_train @ M_train @ Kh0_train
                B_train = Kh0_train @ M_train @ y[train]
                scores[it1].append([])
                for alpha_scale in alpha_scales:
                    alpha = self._get_alpha(delta_train, alpha_scale)
                    # a = np.linalg.pinv(KMK_train + alpha * Kh_train) @ B_train
                    a = np.linalg.lstsq(KMK_train + alpha * Kh_train, B_train, rcond=None)[0]
                    res = y[test]
                    res[index[test]] = y[index[test]] - self._get_kernel_h(X=X_test[index[test],:], Y=X_train, gamma_h=gamma_h) @ a
                    scores[it1][it2].append((res.T @ M_test @ res).reshape(-1)[0] / (res.shape[0]**2))

        #self.alpha_scales = alpha_scales
        avg_scores = np.mean(np.array(scores), axis=0)
        best_ind = np.unravel_index(np.argmin(avg_scores), avg_scores.shape)
        self.gamma_h = gamma_hs[best_ind[0]]
        Kh = self._get_kernel_h(X, gamma_h=self.gamma_h)
        Kh0 = np.zeros_like(Kh)
        Kh0[index,:] = Kh[index,:]

        self.best_alpha_scale = alpha_scales[best_ind[1]]
        delta = self._get_delta(np.sum(index))
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf

        # self.a = np.linalg.pinv(
        #     Kh0 @ M @ Kh0 + self.best_alpha * Kh) @ Kh0 @ M @ y
        self.a = np.linalg.lstsq(
            Kh0 @ M @ Kh0 + self.best_alpha * Kh, Kh0 @ M @ y, rcond=None)[0]
        return self

class ApproxRKHSIV_q(_BaseRKHSIV):

    def __init__(self, kernel_approx='nystrom', n_components=20,
                 gamma_f='auto', gamma_h=0.1,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto'):
        """
        Parameters:
            kernel_approx : what approximator to use; either 'nystrom' or 'rbfsampler' (for kitchen sinks)
            n_components : how many approximation components to use
            # kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma_h : the gamma parameter for the kernel of h
            gamma_f : the gamma parameter for the kernel of f
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h 
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5

    def _get_new_approx_instance(self, gamma):
        if self.kernel_approx == 'rbfsampler':
            return RBFSampler(gamma=gamma, n_components=self.n_components, random_state=1)
        elif self.kernel_approx == 'nystrom':
            return Nystroem(kernel='rbf', gamma=gamma, random_state=1, n_components=self.n_components)
        else:
            raise AttributeError("Invalid kernel approximator")

    def fit(self, X, y, condition, index):
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition=condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=self.gamma_f)
        RootKf = self.featCond.fit_transform(condition)

        # Standardize X and get RootKh
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)
        RootKh[np.logical_not(index),:] = 0

        # delta & alpha
        n = X.shape[0]
        #delta = self._get_delta(n)
        delta = self._get_delta(np.sum(index)) # only sum(index) of effective data
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ y
        # self.a = np.linalg.pinv(W) @ B
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self

    def predict(self, X):
        X = self.transX.transform(X)
        return self.featX.transform(X) @ self.a


class ApproxRKHSIVCV_q(ApproxRKHSIV_q):

    def __init__(self, kernel_approx='nystrom', n_components=25,
                 gamma_f='auto', gamma_hs = 'auto', n_gamma_hs=10,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            kernel_approx : what approximator to use; either 'nystrom' or 'rbfsampler' (for kitchen sinks)
            n_components : how many nystrom components to use
            gamma_f : the gamma parameter for the kernel of f
            gamma_hs : the list of gamma parameters for kernel of h
            n_gamma_hs : how many gamma_hs to try
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how mny alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components

        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs=n_gamma_hs

        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv

    def _get_gamma_hs(self,X):
        if _check_auto(self.gamma_hs):
            params = {"squared": True}
            K_X_euclidean = sklearn.metrics.pairwise_distances(X = X, metric='euclidean', **params)
            # return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)/X.shape[1]
            return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0],-1)], np.array(range(1, self.n_gamma_hs))/self.n_gamma_hs)
        else:
            return self.gamma_hs

    def fit(self, X, y, condition, index):
        # index is a np vector with bool value and the same length of Y, indicate which Y's should be estimate by h
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)

        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition = condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=gamma_f)
        RootKf = self.featCond.fit_transform(condition)

        # Standardize X and get gamma_hs -> RootKhs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        gamma_hs = self._get_gamma_hs(X)
        RootKhs=[]
        for gammah in gamma_hs:
            RootKh = self._get_new_approx_instance(gamma=gammah).fit_transform(X)
            RootKh[np.logical_not(index),:] = 0
            RootKhs.append(RootKh)

        # delta & alpha
        n = X.shape[0]
        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        #delta_train = self._get_delta(n_train)
        #delta_test = self._get_delta(n_test)
        #delta = self._get_delta(n)

        scores = []
        for it1, (train, test) in enumerate(StratifiedKFold(n_splits=self.cv).split(X,index)):
            delta_test = self._get_delta(np.sum(index[test]))
            delta_train = self._get_delta(np.sum(index[train]))
            RootKf_train, RootKf_test = RootKf[train], RootKf[test]
            Q_train = np.linalg.pinv(
                RootKf_train.T @ RootKf_train / (2 * n_train * (delta_train**2)) + np.eye(self.n_components) / 2)
            Q_test = np.linalg.pinv(
                RootKf_test.T @ RootKf_test / (2 * n_test * (delta_test**2)) + np.eye(self.n_components) / 2)
            scores.append([])
            for it2, RootKh in enumerate(RootKhs):
                RootKh_train, RootKh_test = RootKh[train], RootKh[test]
                A_train = RootKh_train.T @ RootKf_train
                AQA_train = A_train @ Q_train @ A_train.T
                B_train = A_train @ Q_train @ RootKf_train.T @ y[train]
                scores[it1].append([])
                for alpha_scale in alpha_scales:
                    alpha = self._get_alpha(delta_train, alpha_scale)
                    # a = np.linalg.pinv(AQA_train + alpha *
                    #                    np.eye(self.n_components)) @ B_train
                    a = np.linalg.lstsq(AQA_train + alpha *
                                       np.eye(self.n_components), B_train, rcond=None)[0]
                    res = RootKf_test.T @ (y[test] - RootKh_test @ a)
                    scores[it1][it2].append((res.T @ Q_test @ res).reshape(-1)[0] / (len(test)**2))

        #self.alpha_scales = alpha_scales
        avg_scores = np.mean(np.array(scores), axis=0)
        best_ind = np.unravel_index(np.argmin(avg_scores), avg_scores.shape)

        self.gamma_h = gamma_hs[best_ind[0]]
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)
        RootKh[np.logical_not(index),:] = 0

        self.best_alpha_scale = alpha_scales[best_ind[1]]
        delta = self._get_delta(np.sum(index))
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + self.best_alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ y
        # self.a = np.linalg.pinv(W) @ B
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self