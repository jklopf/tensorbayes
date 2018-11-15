# Imports
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
tfd = tfp.distributions
from tensorflow.python import debug as tf_debug

'''
This version should be able to retrieve the simulated parameters and
store the history of the sampling, as NumpyBayes_v2.py does. 

The next version will implement the tensorflow dataset API instead of
placeholders to feed data, and will be called:
- TensorBayes_v3.3.py

'''


# sess = tf.Session()
# sess.close()


# a = tf.constant([1,2,3,4,5])
# print_op = tf.print(a)
# with tf.control_dependencies([print_op]):
#     out = tf.add(a, a)
# sess.run(out)






# Start time measures
start_time = time.clock()

# Reset the graph
tf.reset_default_graph()

# Reproducibility
# Seed setting for reproducable research.
# 
# Set numpy seed
np.random.seed(1234)

# Set graph-level seed
tf.set_random_seed(1234)


# Util functions

def tf_squared_norm(vector):
    sum_of_squares = tf.reduce_sum(tf.square(vector))
    return sum_of_squares
    
def np_squared_norm(vector):
    sum_of_squares = np.sum(np.square(vector))
    return sum_of_squares


# ## Distributions functions
# 


# rnorm is defined using the variance (i.e sigma^2)
def rnorm(mean, var): 
    sd = tf.sqrt(var)
    dist = tfd.Normal(loc= mean, scale= sd)
    sample = dist.sample()
    return sample

def rbeta(a, b):
    dist = tfd.Beta(a, b)
    sample = dist.sample()
    return sample

def rinvchisq(df, scale): # scale factor = tau^2
    dist = tfd.Chi2(df)
    sample = (df * scale)/dist.sample()
    return sample

def rbernoulli(p):
    dist = tfd.Bernoulli(probs=p)
    sample = dist.sample()
    return sample


# Sampling functions
# 

# sample mean
def sample_mu(N, Sigma2_e, Y, X, betas):    
    mean = tf.reduce_sum(tf.subtract(Y, tf.matmul(X, betas)))/N
    var = Sigma2_e/N
    sample = rnorm(mean,var)
    return sample

# sample variance of beta
def sample_sigma2_b(betas, NZ, v0B, s0B):
    df = v0B+NZ
    scale = (tf_squared_norm(betas)+v0B*s0B) / df  
    sample = rinvchisq(df, scale)
    return sample


# sample error variance of Y
def sample_sigma2_e(N, epsilon, v0E, s0E):
    df = v0E + N
    scale = (tf_squared_norm(epsilon) + v0E*s0E) / df
    sample = rinvchisq(df, scale)
    return sample

# sample mixture weight
def sample_w(M, NZ):
    sample = rbeta(1 + NZ, 1 + M - NZ)
    return sample

# sample a beta
def sample_beta(x_j, eps, s2e, s2b, w, beta_old):
    eps = eps + (x_j*beta_old)
    Cj = tf_squared_norm(x_j) + s2e/s2b
    rj = tf.tensordot(tf.transpose(x_j), eps, 1)[0,0]
    ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*s2e ))) * tf.sqrt((s2b*Cj)/s2e)
    pij = w / (w + ratio*(1-w))
    pij_text = tf.constant("pij: ")
    pij_print = tf.print(pij_text ,pij)

    with tf.control_dependencies([pij_print]):
        toss = rbernoulli(pij)
    def case_zero():
        return 0., 0. # could return a list [beta,ny]
    def case_one():
        return rnorm(rj/Cj, s2e/Cj), 1. # could return a list [beta,ny]
    beta_new, ny_new = tf.cond(tf.equal(toss,1), case_one, case_zero)
    #beta_new, incl = tf.case([(tf.equal(toss, 0), case_zero)], default=case_one)
    # maybe use tf.cond since we only got 1 pair ?
    # do we handle ny/nz here ?
    eps = eps - (x_j*beta_new)
    return beta_new, ny_new, eps # could return a list [beta,ny]w


## Simulate data

def build_toy_dataset(N, M, var_g):
    
    sigma_b = np.sqrt(var_g/M)
    sigma_e = np.sqrt(1 - var_g)
    
    beta_true = np.random.normal(0, sigma_b , M)
    x = sigma_b * np.random.randn(N, M) 
    y = np.dot(x, beta_true) + np.random.normal(0, sigma_e, N)
    return x, y, beta_true



# Simulated data parameters

N = 5000       # number of data points
M = 10        # number of features
#N = tf.constant(n) # tensor equivalent
#M = tf.constant(m) # tensor equivalent
var_g = 0.7   # M * var(Beta_true)
              # Var(Beta_true) = Var(g) / M
              # Var(error) = 1 - Var(g) 

x, y, beta_true = build_toy_dataset(N, M, var_g)
X = tf.constant(x, shape=[N,M], dtype=tf.float32, name='X')
Y = tf.constant(y, shape=[N,1], dtype=tf.float32, name='Y')

# Could be implemented:
# building datasets using TF API without numpy


# # Precomputations - not practicable with huge datasets
# sm = np.zeros(M)
# for i in range(M):
#     sm[i] = np_squared_norm(x[:,i])


'''
TODO: 	Actually implement all the algorithm optimizations of the reference article
        which are not implemented here. Depends on later implementations of input pipeline.
'''

#  Parameters setup
#
# Distinction between constant and variables
# Variables: values might change between evaluation of the graph
# (if something changes within the graph, it should be a variable)

# Variables:

Emu = tf.Variable(0., dtype=tf.float32, name='Emu')
Ebeta = tf.Variable(tf.zeros([M,1], dtype=tf.float32), dtype=tf.float32, name='Ebeta')
Ny = tf.Variable(tf.zeros(M, dtype=tf.float32), dtype=tf.float32, name='Ny')
NZ = tf.Variable(0., dtype=tf.float32, name='NZ')
Ew = tf.Variable(0., dtype=tf.float32, name='Ew')
epsilon = tf.Variable(Y, dtype=tf.float32, name='epsilon')

s2e_initial_value = np_squared_norm(y) / (N*0.5)
s2b_initial_value = np.random.beta(1,1)
Sigma2_e = tf.Variable(s2e_initial_value, dtype=tf.float32, name='Sigma2_e')
Sigma2_b = tf.Variable(s2b_initial_value, dtype=tf.float32, name='Sigma2_b')

# Constants:

vEmu = tf.ones([N,1], dtype=tf.float32, name='vEmu')
v0E = tf.constant(0.001, dtype=tf.float32, name='v0E')
v0B = tf.constant(0.001, dtype=tf.float32, name='v0B')
s0E = tf.constant(s2e_initial_value / 2, dtype=tf.float32, name='s0E')
s0B = tf.constant(s2b_initial_value / 2, dtype=tf.float32, name='s0B')

# Placeholders:
Xj = tf.placeholder(tf.float32, shape=(N,1), name='col_ph')
ind = tf.placeholder(tf.int32, shape=(), name='ind_ph')


# Print stuff:
# TODO: construct the op with tf.print() so that values get automatically printed

print_dict = {'Emu': Emu, 'Ew': Ew, 
              'NZ': NZ, 'Sigma2_e': Sigma2_e,
              'Sigma2_b': Sigma2_b}


# Tensorboard graph
# TODO: look up what TensorBoard can do, this can be used in the end to have a graph representation of the algorithm.
# Also, for graph clarity, operations should be named.

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())


# Computations
ta_beta, ta_ny, ta_eps = sample_beta(Xj, epsilon, Sigma2_e, Sigma2_b, Ew, Ebeta[ind,0]) # Ebeta[ind] might be replaced by using dictionnaries key/value instead.
ta_epsilon = Y - tf.matmul(X,Ebeta) - vEmu*Emu
ta_s2b = sample_sigma2_b(Ebeta,NZ,v0B,s0B)
ta_s2e = sample_sigma2_e(N,epsilon,v0E,s0E)
ta_nz = tf.reduce_sum(Ny)
ta_ew = sample_w(M, NZ)


# Print ops
beta_ny_text = tf.constant("beta ; ny:")
print_op = tf.print(beta_ny_text,ta_beta, ta_ny)

# Assignment ops
# As we don't chain assignment operations, assignment does not require to return the evaluation of the new value
# therefore, all read_value are set to False. No idea if this changes anything.

emu_up = Emu.assign(sample_mu(N, Sigma2_e, Y, X, Ebeta), read_value=False)
beta_item_assign_op = Ebeta[ind,0].assign(ta_beta) 		# when doing item assignment, read_value becomes an unexpected parameter, 
ny_item_assign_op = Ny[ind].assign(ta_ny) 			# as tensorflow doesn't know what to return the single item or the whole variable
nz_up = NZ.assign(ta_nz, read_value=False)
eps_up_fl = epsilon.assign(ta_eps, read_value=False)
eps_up = epsilon.assign(ta_epsilon, read_value=False)
ew_up = Ew.assign(ta_ew, read_value=False)
s2b_up = Sigma2_b.assign(ta_s2b, read_value=False)
s2e_up = Sigma2_e.assign(ta_s2e, read_value=False)

with tf.control_dependencies([print_op]):
    up_grp = tf.group(beta_item_assign_op, ny_item_assign_op, eps_up)





'''
Run with `read_value = True`: 63.4s
Run with `read_value = False`: 62.2s
'''


# Sampling log operations



# Number of Gibbs sampling iterations
num_iter = 5

sess = tf.Session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "wl5s-253-236.unil.ch:6064")



# Initialize variable
sess.run(tf.global_variables_initializer())

# Gibbs sampler iterations
for i in range(num_iter):
    print("Gibbs sampling iteration: ", i)
    sess.run(emu_up)
    #sess.run(ny_reset)
    index = np.random.permutation(M)

    for marker in index:
        current_col = x[:,[marker]]
        feed = {ind: marker, Xj: current_col}
        sess.run(up_grp, feed_dict=feed)
    sess.run(nz_up)
    sess.run(ew_up)
    sess.run(eps_up)
    sess.run(s2b_up)
    sess.run(s2e_up)
    # Print operations 
    print(sess.run(print_dict))

sess.close()
