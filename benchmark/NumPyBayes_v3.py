#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:22:40 2018

@author: Jonathan Klopfenstein
"""

import numpy as np
from tqdm import tqdm
from timeit import timeit



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
    
    sigma_b = np.sqrt(var_g/M)
    sigma_e = np.sqrt((1 - var_g))
    
    beta_true = np.random.normal(0, sigma_b , M)
    x = sigma_b * np.random.randn(N, M)
    #x=preprocessing.scale(x)
    y = np.dot(x, beta_true) + np.random.normal(0, sigma_e, N)
    return x, y, beta_true

# Parameters of simulated data
    


def gibb():
    N = 5000
    M = 10
    var_g=0.7

    # var(b) = var(g) / M
    # var(e) = 1 - var(g)
    # var(y) = var(g) + var(e)

    x, y, beta_true = build_toy_dataset(N,M,var_g)

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
    s0B = sigma2_b / 2
    s0E = sigma2_e / 2

    # Precomputations

    ###############################################################################


    # Gibbs sampling iterations


    num_iter = 5000

    sigma_e_log = []
    sigma_b_log = []
    beta_log = []
    ny_log = []

    print('\n', 'Begin Gibbs sampling', '\n')
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
        ny_log.append(ny)
        Ew = sample_w(M, NZ)
        sigma2_b = sample_sigma2_b(Ebeta, NZ, v0B, s0B)
        sigma_b_log.append(sigma2_b)
        sigma2_e = sample_sigma2_e(N, epsilon, v0E, s0E)
        sigma_e_log.append(sigma2_e)
                
        if(i > 2000):
            beta_log.append(Ebeta.reshape(M))
    print('\n','Results:')
    print(
        "mean Ebeta",
        'posterior inclusion probability',
        ' beta_true',
        sep='\t' + '\t')
    for i in range(M):
        print(
            np.round(np.mean(beta_log, axis = 0).reshape(M,1)[i],5),
            np.mean(ny_log, axis = 0).reshape(M,1)[i],
            beta_true[i],
            sep='\t' + '\t')

    print(" ")
    print("mean sigma2_e:", np.round(np.mean(sigma_e_log[2500:5000]), 5))
    print("mean sigma2_b:", np.round(np.mean(sigma_b_log[2500:5000]), 5))

    

gibb()
n_time = 1
cpu_time = np.round(timeit('gibb()', number=n_time, setup='from __main__ import gibb'), 4)
print('Mean runtime (s): {}'.format(cpu_time/(n_time + 1)))


# # Print results
        
# print("mean Ebeta" +  "\t" + "     ", '   ny' + '\t'+ ' beta_true')
# for i in range(M):
#     print(np.round(np.mean(beta_log, axis = 0).reshape(M,1)[i],5), "\t"  + "", ny[i], "\t", beta_true[i])

# print(" ")
# print("mean sigma2_e:" + str(np.mean(sigma_e_log[2500:5000])))
# print("mean sigma2_b:" + str(np.mean(sigma_b_log[2500:5000])))







