# -*- coding: utf-8 -*-
"""
Script for development of benchmark the dirac spike model using NumPy.
This script takes 2 arguments:
N:              Number of individuals
M:              Number of covariates

Everything is sent to stdout.
"""
import numpy as np
from timeit import repeat
import psutil
from tqdm import tqdm
import argparse
from sklearn import preprocessing


# Dev
np.set_printoptions(formatter={'float_kind':'{:0.5f}'.format})

# Reproducibility
np.random.seed(1234)


# Parameters of simulated data

# var(b) = var(g) / M
# var(e) = 1 - var(g)
# var(y) = var(g) + var(e) 

# Set N, M to be pass as arguments when running the script

parser = argparse.ArgumentParser(
    description='Run a simulation study of penalized regression. \
    Please provide the number of individuals, number of covariates and number of script iteration.')
parser.add_argument('n', metavar='N', type=int,
                    help='number of individuals')
parser.add_argument('m', metavar='M', type=int,
                    help='number of covariates')
#parser.add_argument('n_time', metavar='n_time', type=int,
#                    help='number of script iteration')
args = parser.parse_args()

N = args.n     # Number of individuals
M = args.m     # Number of covariates
var_g = 0.7   # Genetic variance


# Benchmark parameters and logs
oa_mean_s2b = []
oa_mean_s2e = []
oa_cor = []
oa_pip = []

# Gibbs sampler function
def gibbs():
    global oa_mean_s2b
    global oa_mean_s2e
    global oa_cor, oa_pip
    global N, M

    ###############################################################################

    # Distribution functions
    def rinvchisq(df, scale):
        sample = (df * scale)/(np.random.chisquare(df))
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

    # Data simulation function
    def build_toy_dataset(N, M, var_g):
        sigma_b = np.sqrt(var_g/M)
        sigma_e = np.sqrt((1 - var_g))
        beta_true = np.random.normal(0, sigma_b , M)
        x = sigma_b * np.random.randn(N, M)
        x = preprocessing.scale(x)
        y = np.dot(x, beta_true) + np.random.normal(0, sigma_e, N)
        return x, y, beta_true

    ###############################################################################

    # Simulated data
    x, y, beta_true = build_toy_dataset(N,M,var_g)

    # Parameters setup
    Ebeta = np.zeros(M)
    ny = np.zeros(M)
    Ew = 0.5
    epsilon = y
    NZ = np.zeros(1)
    sigma2_e = squared_norm(y) / (N*0.5)
    sigma2_b = rbeta(1.0, 1.0)
    v0E = 4.0
    v0B = 4.0
    s0B = sigma2_b / 2
    s0E = sigma2_e / 2

    # Dev
    #print('v0E, v0B = {}'.format(v0B))

    # Gibbs sampling iterations parameters and sampling logs
    num_iter = 5000
    burned_samples_threshold = 2000
    sigma_e_log = []
    sigma_b_log = []
    beta_log = []
    ny_log = []

    ###############################################################################

    # Gibbs sampler
    for i in tqdm(range(num_iter)):
        
        # Random order of column
        index = np.random.permutation(M)
        
        # Loop through the column to sample the regression coefficient (betas)
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
        
        # Update the rest of the variables
        NZ = np.sum(ny)
        Ew = sample_w(M, NZ)
        sigma2_b = sample_sigma2_b(Ebeta, NZ, v0B, s0B)
        sigma2_e = sample_sigma2_e(N, epsilon, v0E, s0E)

        # Store sampling logs      
        if(i >= burned_samples_threshold):
            sigma_e_log.append(sigma2_e)
            sigma_b_log.append(sigma2_b)
            ny_log.append(np.copy(ny))
            beta_log.append(np.copy(Ebeta))
        
    
    # Store local results
    mean_ebeta = np.mean(beta_log, axis = 0)
    pip = np.mean(ny_log, axis = 0)
    mean_s2e = np.mean(sigma_e_log)
    mean_s2b = np.mean(sigma_b_log)
    corr_ebeta_betatrue = np.corrcoef(mean_ebeta, beta_true)[0][1]

    # Dev: print the estimated betas, the PiP and the true betas
    dash = '-' * 40
    print(dash)
    print('{:<10s}{:>4s}{:>12s}'.format('Computed betas','PiP','True betas'))
    print(dash)
    for i in range(M):
        print('{:<12.5f}{:>6.3f}{:>12.5f}'.format(mean_ebeta[i], pip[i], beta_true[i]))
    
    # Store overall results
    oa_mean_s2e.append(mean_s2e)
    oa_mean_s2b.append(mean_s2b)
    oa_pip.append(len([num for num in pip if num >= 0.95]))
    oa_cor.append(corr_ebeta_betatrue)

# Measure running times and execute the code n_time
n_time = 1
oa_time = np.round(repeat('gibbs()',repeat=n_time, number=1, setup='from __main__ import gibbs'), 4)

# Measure memory usage
mem = psutil.Process().memory_info()
rss = mem.rss / (1024**2)
vms = mem.vms / (1024**2)

# Output benchmark logs
print('\nBenchmarking results: NumPyBayes v3')
print('N = {}, M = {}, var(g) = {}'.format(N,M,var_g))
print('\nMemory usage')
print('rss memory (physical): {} MiB'.format(rss))
print('vms memory (virtual): {} MiB'.format(vms))
print('\nTiming results')
print('Minimal time of execution: {}s'.format(oa_time.min()))
print('Mean time of execution memory: {}s'.format(np.mean(oa_time)))

# Write results to a .csv
# Order: s2e | s2b | cor | pip | time
results = np.stack((
    oa_mean_s2e,
    oa_mean_s2b,
    oa_cor,
    oa_pip,
    oa_time), axis=-1)

#np.savetxt(
#    'dev.csv',
#    results,
#    delimiter=',',
#    header='sigma2_e, sigma2_b, cor(eb,bt), PiP, time',
#    fmt='%.8f')


print('\nResults:')
dash = '-' * 40
print(dash)
print('{:<10s}{:>7s}{:>9s}{:>4s}{:>6s}'.format('s2e', 's2b     ','corr(eb,bt)', 'PiP','time'))
print(dash)
print(results)







