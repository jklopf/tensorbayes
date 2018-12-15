#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:22:40 2018

@author: Jonathan Klopfenstein
"""

import numpy as np
from timeit import repeat
import psutil
from tqdm import tqdm


# Reproducibility
np.random.seed(1234)

# Sampling functions
def rinvchisq(df, scale):
    sample = (df * scale)/(np.random.chisquare(df) + 1e-8)
    return sample


def rnorm(mean, var):
    # rnorm is defined using the variance (i.e sigma^2)
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
    sigma_b = np.sqrt(var_g/m)
    sigma_e = np.sqrt((1 - var_g))
    beta_true = np.random.normal(0, sigma_b , M)
    x = sigma_b * np.random.randn(N, M)
    #x=preprocessing.scale(x)
    y = np.dot(x, beta_true) + np.random.normal(0, sigma_e, N)
    return x, y, beta_true


# Parameters of simulated data

# var(b) = var(g) / M
# var(e) = 1 - var(g)
# var(y) = var(g) + var(e) 

N = 5000     # Number of individuals
M = 100     # Number of covariates
var_g = 0.7   # Genetic variance


# Benchmark parameters and logs
oa_mean_s2b = []
oa_mean_s2e = []
oa_cor = []
oa_pip = []
oa_time = []

def gibb():
    global oa_mean_s2b
    global oa_mean_s2e
    global oa_cor
    global N, M

    # Simulated data
    x, y, beta_true = build_toy_dataset(N,M,var_g)

    # Parameters setup
    Ebeta = np.zeros(M)
    ny = np.zeros(M)
    Ew = np.zeros(1)
    epsilon = y
    NZ = np.zeros(1)
    sigma2_e = squared_norm(y) / (N*0.5)
    sigma2_b = rbeta(1,1)
    v0E, v0B = 0.001,0.001
    s0B = sigma2_b / 2
    s0E = sigma2_e / 2

    # Gibbs sampling iterations and sampling logs
    num_iter = 5000
    sigma_e_log = []
    sigma_b_log = []
    beta_log = []
    ny_log = []

    for i in tqdm(range(num_iter)):
        
        index = np.random.permutation(M)
        
        for marker in index:
            epsilon = epsilon + x[:,marker] * Ebeta[marker]
            Cj = squared_norm(x[:,marker]) + sigma2_e/sigma2_b
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
        sigma2_b = sample_sigma2_b(Ebeta, NZ, v0B, s0B)
        sigma2_e = sample_sigma2_e(N, epsilon, v0E, s0E)

        # Store sampling logs      
        if(i >= 2000):
            sigma_e_log.append(sigma2_e)
            sigma_b_log.append(sigma2_b)
            ny_log.append(ny)
            beta_log.append(Ebeta.reshape(M))
        
    
    # Store local results
    mean_ebeta = np.mean(beta_log, axis = 0)
    pip = np.mean(ny_log, axis = 0)
    mean_s2e = np.mean(sigma_e_log)
    mean_s2b = np.mean(sigma_b_log)
    corr_ebeta_betatrue = np.corrcoef(mean_ebeta, beta_true)[0][1]

    # Store overall results
    oa_mean_s2e.append(mean_s2e)
    oa_mean_s2b.append(mean_s2b)
    oa_pip.append(len([num for num in pip if num >= 0.95]))
    oa_cor.append(corr_ebeta_betatrue)

# Measure running times and execute the code n_time
n_time = 1
oa_time = np.round(repeat('gibb()',repeat=n_time, number=1, setup='from __main__ import gibb'), 4)

# Measure memory usage
mem = psutil.Process().memory_info()
rss = mem.rss / (1024**2)
vms = mem.vms / (1024**2)

# Output benchmark logs
print('D1 Logs')
print('N = {}, M = {}, var(g) = {}'.format(N,M,var_g))
print('\nrss memory (physical): {} MiB'.format(rss))
print('vms memory (virtual): {} MiB'.format(vms))
print('\nMin time of execution: ', oa_time.min())
print('Mean time of execution memory: ', np.mean(oa_time))

# Write results to a .csv
# Order: s2e | s2b | cor | pip | time
results = np.stack((
    oa_mean_s2e,
    oa_mean_s2b,
    oa_cor,
    oa_pip,
    oa_time), axis=-1)

#np.savetxt('n1.csv', results, delimiter=',', header='s2e,s2b,corr(eb,bt),PiP,time')



# Dev
np.set_printoptions(formatter={'float_kind':'{:0.5f}'.format})

print('\n Computed betas \t \t Expected betas')
for i in range(M):
    print(mean_ebeta[i],beta_true[i], sep='\t\t')


print('\nResults:')
print('s2e    |    s2b    |    cor    |    pip    |    time')
print(results)







