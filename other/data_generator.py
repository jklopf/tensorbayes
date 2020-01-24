import numpy as np
import csv

# Set seed for reproducible research
np.random.seed(1234)

# Simulate data
def build_toy_dataset(N, M, var_g):
    
    sigma_b = np.sqrt(var_g/M)
    sigma_e = np.sqrt(1 - var_g)
    beta_true = np.random.normal(0, sigma_b , M)
    x = sigma_b * np.random.randn(N, M) 
    y = np.dot(x, beta_true) + np.random.normal(0, sigma_e, N)
    return x, y, beta_true.reshape(M,1)

# Simulated data parameters
'''
Var(b) = Var(g) / M
Var(e) = 1 - Var(g)
'''

N = 5000       # number of data points
M = 10        # number of features
var_g = 0.7   # genetic variance parameter

x, y, beta_true = build_toy_dataset(N, M, var_g)
x = np.transpose(x)
y = np.transpose(y.reshape([N,1]))
beta_true = np.transpose(beta_true)

# Write to disk in a .csv file
np.savetxt('x.csv', x, delimiter=',')
np.savetxt('y.csv', y, delimiter=',')
np.savetxt('b_true.csv', beta_true, delimiter=',')