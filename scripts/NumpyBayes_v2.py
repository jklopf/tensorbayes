#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:22:40 2018

@author: Jonathan Klopfenstein
"""

import numpy as np
#from scipy.stats import invgamma
import time
from sklearn import preprocessing


# Timer
start_time = time.clock()

# Reproducibility
#np.random.seed(123)

###############################################################################

# Distribution functions

# =============================================================================
# def rinvchisq(df, scale):
#     a = df * 0.5
#     b = df * scale * 0.5
#     return invgamma.rvs(a, scale=b)
# =============================================================================
   
#def rinvchisq(df, scale):
#    return 1.0 / np.random.gamma(df/2.0, df * scale / 2.0)

def rinvchisq(df, scale):
    sample = (df * scale)/np.random.chisquare(df)
    return sample



# rnorm is defined using the variance (i.e sigma^2)
def rnorm(mean, var):
    sd = np.sqrt(var)
    return np.random.normal(mean, sd)

def rbeta(a,b):
    return np.random.beta(a,b)

def rbernouilli(p):
    return np.random.binomial(1, p)

###############################################################################

# Util functions
    
def squared_norm(vector):
    return np.sum(np.square(vector))

###############################################################################


# Sampling functions
    
def sample_mu(N, sigma2_e, Y, X, beta):
    mean = np.sum(Y - np.matmul(X,beta))/N
    var = sigma2_e/N
    return rnorm(mean, var)

def sample_sigma2_e(N, epsilon, v0E, s0E):
    df = v0E + N
    scale = (squared_norm(epsilon) + v0E*s0E)/df
    sample = rinvchisq(df, scale)
    return sample

def sample_sigma2_b(beta, NZ, v0B, s0B):
    df = v0B + NZ
    scale = (squared_norm(beta) + v0B*s0B)/df # * NZ or not ????
    sample = rinvchisq(df, scale)
    return sample

def sample_w(M, NZ):
    sample = rbeta(1 + NZ, 1 + M - NZ) 
    return sample

###############################################################################

# Data simulation
    
def build_toy_dataset(N, M, var_g):
    
    sigma_b = np.sqrt(var_g/M)
    sigma_e = np.sqrt((1 - var_g))
    
    beta_true = np.random.normal(0, sigma_b , M)
    x = sigma_b * np.random.randn(N, M)
    x=preprocessing.scale(x)
    y = np.dot(x, beta_true) + np.random.normal(0, sigma_e, N)
    return x, y, beta_true

# Parameters of simulated data
    
N = 5000
M = 10
var_g=0.7

# var(b) = var(g) / M
# var(e) = 1 - var(g)
# var(y) = var(g) + var(e)

x, y, beta_true = build_toy_dataset(N,M,var_g)



#x = preprocessing.scale(x)

#y = preprocessing.scale(y)
#x= preprocessing.scale(x)
#y=preprocessing.scale(y)
# beta_true = np.linspace(-4.,10.,10)

#beta_true = np.ones(M) * 0.25
#x = np.random.randn(N,M)
#y = np.matmul(x, beta_true) +  (np.random.randn(N) * 0.375)

###############################################################################

# Parameters setup

Emu = np.zeros(1)
vEmu = np.ones(N)
Ebeta = np.zeros(M)
ny = np.zeros(M)
Ew = np.zeros(1)
epsilon = y
NZ = np.zeros(1)
sigma2_e = squared_norm(y) / (N*0.5)
sigma2_b = rbeta(1,1)

v0E, v0B = 0.001,0.001
#v0E, v0B = 4,4
s0B = sigma2_b / 2
s0E = sigma2_e / 2
#s0B=0.01
#s0E=0.01
#s0B, s0E = 0.001, 0.001
###############################################################################

# Precomputations

sm = np.zeros(M)
for i in range(M):
    sm[i] = squared_norm(x[:,i])

###############################################################################


# Gibbs sampling iterations


num_iter = 5000

sigma_e_log = []
sigma_b_log = []
beta_log = []

for i in range(num_iter):
    
    time_in = time.clock()
    Emu = sample_mu(N, sigma2_e, y, x, Ebeta)
    index = np.random.permutation(M)
    print("Gibbs sampling iteration:", i)
    #print("Current marker:")
    
    for marker in index:
        #print(marker, end=" ", flush=True)
        epsilon = epsilon + x[:,marker] * Ebeta[marker]
        Cj = sm[marker] + sigma2_e/sigma2_b
        rj = np.dot(x[:,marker], epsilon)
        ratio = np.sqrt((sigma2_b * Cj)/sigma2_e)*np.exp(-(np.square(rj)/(2*Cj*sigma2_e)))
        pij = Ew/(Ew + ratio*(1-Ew))
        ny[marker] = rbernouilli(pij)
        if (ny[marker] == 0):
            
            Ebeta[marker] = 0
            
        elif (ny[marker] == 1):
            
            Ebeta[marker] = rnorm(rj/Cj, sigma2_e/Cj)
        
        epsilon = epsilon - x[:,marker] * Ebeta[marker]
    
    
    NZ = np.sum(ny)
    Ew = sample_w(M, NZ)
    epsilon = y - np.matmul(x,Ebeta) - vEmu*Emu
    sigma2_b = sample_sigma2_b(Ebeta, NZ, v0B, s0B)
    sigma_b_log.append(sigma2_b)
    sigma2_e = sample_sigma2_e(N, epsilon, v0E, s0E)
    sigma_e_log.append(sigma2_e)
    
    # Print ops
    time_out = time.clock()
    elapsed_time = time_out - time_in
    
    if(i>4994):
        print("")
        print("Emu: {}, Ew: {}, NZ: {}, sigma2_e: {}, sigma2_b: {}".format(
                round(Emu,5),round(Ew,5),NZ, round(sigma2_e,5), round(sigma2_b,5)))
        print("")
        print("Time for the {}th generation: {}".format(i, elapsed_time))
        print("")
        
    if(i > 2000):
        beta_log.append(Ebeta.reshape(M))
        
    
    
    
    
        
print("mean Ebeta" +  "\t" + "     ", '   ny' + '\t'+ ' beta_true')
for i in range(M):
    print(np.round(np.mean(beta_log, axis = 0).reshape(M,1)[i],5), "\t"  + "", ny[i], "\t", beta_true[i])

total_time =   time.clock()-start_time
print("Total time: " + str(total_time) + "s")
print(" ")
print("mean sigma2_e:" + str(np.mean(sigma_e_log[2500:5000])))
print("mean sigma2_b:" + str(np.mean(sigma_b_log[2500:5000])))







