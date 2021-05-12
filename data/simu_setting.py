#%%
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import uniform
from scipy.stats import randint
import pandas as pd

class prox_data:
    def __init__(self, samp_size: int, add_noise, seed = None):
        """
        samp_size : sample size n
        add_noise     : True or False, add noise to Y or not
        """
        self.add_noise = add_noise
        if seed is None:
            self.seed = [None, None, None, None, None]
        else:
            self.seed = randint.rvs(0, 4294967295, size=5, random_state=seed) #seed must between 0 and 2**32-1

        # generate X
        self.X = multivariate_normal.rvs(mean = [0.25, 0.25], cov = [[0.0625,0],[0,0.0625]], size = samp_size, random_state=self.seed[0])
        # generate A
        self.A = (uniform.rvs(size=samp_size, random_state=self.seed[1]) - 1./(1. + np.exp(self.X @ [0.125, 0.125])) < 0).astype(int)

        # Add irrelevant X
        self.X_irre = multivariate_normal.rvs(mean = -0.25, cov=4, size = samp_size, random_state=self.seed[2])
    
        # generate Z,W,U|A,X
        ZWU = multivariate_normal.rvs(size=samp_size, cov = [[1,0.25,0.5], [0.25,1,0.5], [0.5,0.5,1]], random_state=self.seed[3])
        mean_coef = np.concatenate((np.ones((samp_size,1)),self.A.reshape(-1,1), self.X),axis=1)
        ZWU = ZWU + mean_coef @ np.array([[0.25,0.25,0.25],[0.25,0.125,0.25],[0.25,0.25,0.25],[0.25,0.25,0.25]])
        self.Z = ZWU[:,0]
        self.W = ZWU[:,1]
        self.U = ZWU[:,2]
        # calculate q0(Z,A,X)
        self.q0 = self.A*(1.+np.exp(0.25 - 0.5*self.Z - 0.125*self.A + 0.25*self.X[:,0]+0.25*self.X[:,1]))\
            + (1-self.A)*(1.+np.exp(-0.25 + 0.5*self.Z + 0.125*self.A - 0.25*self.X[:,0]-0.25*self.X[:,1]))
        return

    def h0_linear_fun(self,W,A,X0,X1):
        return 2. + 0.75*A + 5.*W + 0.25*X0 + 0.25*X1 + A*(2.*W-4.*X0-3.*X1)
    
    def h0_linear_X_fun(self,W,A,X0,X1):
        return 2. + 0.5*A + 8.*W + 0.25*X0 + 0.25*X1 + A*(3.*X0-5.*X1)

    def h0_nonlinear_fun(self,W,A,X0,X1):
        return 2. + 2.3*A + 4.*W + X0**2 + X1**2 \
        + A*(-2.5*W+np.abs(X0-1)-np.abs(X1+1) + W*(np.sin(X0)-2*np.cos(X1)))

    def h0_nonlinear_X_fun(self,W,A,X0,X1):
        return 2. + 0.25*A + 5.*W + X0**2 + X1**2 + A*(-6*X0*X1)

    def q0_fun(self,Z,A,X0,X1):
        return 1. + np.exp((-1)**(1-A)*(0.25-0.5*Z-0.125*A+0.25*X0+0.25*X1))

    def gen_Y(self, linearity, indepOutProx, Xonly): # Y is self.data[:,8]
        """
        linearity    : string {"linear","nonlinear"}
        indepOutProx : string {"WindepY","WdepY"}
        Xonly        : string {"XW", "X"}  #h(X,1,W) - h(X,0,W) depends on X,W or only depends on X
        """
        self.linearity = linearity       #"linear" or "nonlinear"
        self.indepOutProx = indepOutProx #"WindepY" or "WdepY"
        omega = 0 if indepOutProx=="WindepY" else 2

        if linearity=="linear":
            if Xonly=="XW":
                # h0
                self.h0 = self.h0_linear_fun(self.W,self.A,self.X[:,0],self.X[:,1])
                self.h0_a0 = self.h0_linear_fun(self.W,np.zeros_like(self.A),self.X[:,0],self.X[:,1])
                self.h0_a1 = self.h0_linear_fun(self.W, np.ones_like(self.A),self.X[:,0],self.X[:,1])

                # Y
                self.Y = 2.+0.75*self.A+0.25*self.X[:,0]+0.25*self.X[:,1]\
                    -4.*self.A*self.X[:,0]-3.*self.A*self.X[:,1]+omega*self.W\
                        +(5.+2.*self.A-omega)*(0.25+0.25*self.X[:,0]+0.25*self.X[:,1] + 0.5*(self.U-0.25-0.25*self.X[:,0]-0.25*self.X[:,1]))
                if self.add_noise:
                    self.Y = self.Y + uniform.rvs(loc=-1., scale=2., size=self.A.shape[0], random_state=self.seed[4]) # unif[-2,2] random error
                # global optimal regime
                self.GOR = np.sign(0.75 -4.*self.X[:,0] -3.*self.X[:,1]\
                        + 2*(0.25+0.25*self.X[:,0]+0.25*self.X[:,1]+0.5*(self.U-0.25-0.25*self.X[:,0]-0.25*self.X[:,1])))
            else:
                # h0
                self.h0 = self.h0_linear_X_fun(self.W,self.A,self.X[:,0],self.X[:,1])
                self.h0_a0 = self.h0_linear_X_fun(self.W,np.zeros_like(self.A),self.X[:,0],self.X[:,1])
                self.h0_a1 = self.h0_linear_X_fun(self.W, np.ones_like(self.A),self.X[:,0],self.X[:,1])

                # Y
                self.Y = 2.+0.5*self.A+0.25*self.X[:,0]+0.25*self.X[:,1]\
                    +3.*self.A*self.X[:,0]-5.*self.A*self.X[:,1]+omega*self.W\
                        +(8.-omega)*(0.25+0.25*self.X[:,0]+0.25*self.X[:,1] + 0.5*(self.U-0.25-0.25*self.X[:,0]-0.25*self.X[:,1]))
                if self.add_noise:
                    self.Y = self.Y + uniform.rvs(loc=-1., scale=2., size=self.A.shape[0], random_state=self.seed[4]) # unif[-2,2] random error
                # global optimal regime
                self.GOR = np.sign(0.75 +3.*self.X[:,0] -5.*self.X[:,1])
        else:
            if Xonly=="XW":
                # h0
                self.h0 = self.h0_nonlinear_fun(self.W,self.A,self.X[:,0],self.X[:,1])
                self.h0_a0 = self.h0_nonlinear_fun(self.W,np.zeros_like(self.A),self.X[:,0],self.X[:,1])
                self.h0_a1 = self.h0_nonlinear_fun(self.W, np.ones_like(self.A),self.X[:,0],self.X[:,1])

                # Y
                self.Y = 2.+2.3*self.A+self.X[:,0]**2+self.X[:,1]**2\
                    +self.A*(np.abs(self.X[:,0]-1)-np.abs(self.X[:,1]+1))+omega*self.W\
                        +(4.-2.5*self.A+self.A*(np.sin(self.X[:,0])-2*np.cos(self.X[:,1]))-omega)\
                            *(0.25+0.25*self.X[:,0]+0.25*self.X[:,1] + 0.5*(self.U-0.25-0.25*self.X[:,0]-0.25*self.X[:,1]))
                if self.add_noise:
                    self.Y = self.Y + uniform.rvs(loc=-1., scale=2., size=self.A.shape[0], random_state=self.seed[4]) # unif[-2,2] random error
                # global optimal regime
                self.GOR = np.sign(2.3+np.abs(self.X[:,0]-1)-np.abs(self.X[:,1]+1)\
                    + (-2.5+np.sin(self.X[:,0])-2*np.cos(self.X[:,1]))\
                        *(0.25+0.25*self.X[:,0]+0.25*self.X[:,1]+0.5*(self.U-0.25-0.25*self.X[:,0]-0.25*self.X[:,1])))
            else:
                # h0
                self.h0 = self.h0_nonlinear_X_fun(self.W,self.A,self.X[:,0],self.X[:,1])
                self.h0_a0 = self.h0_nonlinear_fun(self.W,np.zeros_like(self.A),self.X[:,0],self.X[:,1])
                self.h0_a1 = self.h0_nonlinear_fun(self.W, np.ones_like(self.A),self.X[:,0],self.X[:,1])

                # Y
                self.Y = 2.+0.25*self.A+self.X[:,0]**2+self.X[:,1]**2\
                    +self.A*(-6*self.X[:,0]*self.X[:,1])+omega*self.W\
                        +(5.-omega)\
                            *(0.25+0.25*self.X[:,0]+0.25*self.X[:,1] + 0.5*(self.U-0.25-0.25*self.X[:,0]-0.25*self.X[:,1]))
                if self.add_noise:
                    self.Y = self.Y + uniform.rvs(loc=-1., scale=2., size=self.A.shape[0], random_state=self.seed[4]) # unif[-2,2] random error
                # global optimal regime
                self.GOR = np.sign(0.25-6*self.X[:,0]*self.X[:,1])

        df = pd.DataFrame()
        df["A"]  = self.A
        df["X0"] = self.X[:,0]
        df["X1"] = self.X[:,1]
        df["X2"] = self.X_irre # irrelevent X
        df["Z"]  = self.Z
        df["W"]  = self.W
        df["Y"]  = self.Y
        df["U"]  = self.U
        df["h0"]  = self.h0
        df["q0"]  = self.q0
        df["h0_a0"]  = self.h0_a0
        df["h0_a1"]  = self.h0_a1
        df["GOR"]  = self.GOR
        return df

