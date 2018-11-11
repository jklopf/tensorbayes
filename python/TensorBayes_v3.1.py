# Imports
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
tfd = tfp.distributions

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

def rbeta(alpha, beta):
    dist = tfd.Beta(alpha, beta)
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
    sample = rbeta(NZ+1, M-NZ+1)
    return sample

# sample a beta
def sample_beta(x_j, eps, s2e, s2b, w, beta_old):
    eps = eps + (x_j*beta_old)
    Cj = tf_squared_norm(x_j) + s2e/s2b
    rj = tf.tensordot(tf.transpose(x_j), eps, 1)[0][0]
    ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*s2e ))) * tf.sqrt((s2b*Cj)/s2e)
    pij = w / (w + ratio*(1-w))
    toss = rbernoulli(pij)
    def case_zero(): return 0., 0 # could return a list [beta,ny]
    def case_one(): return rnorm(rj/Cj, s2e/Cj), 1 # could return a list [beta,ny]
    beta_new, incl = tf.case([(tf.not_equal(toss, 0), case_one)], default=case_zero)
    # do we handle ny/nz here ?
    eps = eps - (x_j*beta_new)
    return beta_new, incl, eps # could return a list [beta,ny]


## Simulate data

def build_toy_dataset(N, M, var_g):
    
    sigma_b = np.sqrt(var_g/M)
    sigma_e = np.sqrt(1 - var_g)
    
    beta_true = np.random.normal(0, sigma_b , M)
    x = sigma_b * np.random.randn(N, M) 
    y = np.dot(x, beta_true) + np.random.normal(0, sigma_e, N)
    return x, y, beta_true



# Simulated data parameters

N = 100       # number of data points
M = 10        # number of features
var_g = 0.7   # M * var(Beta_true)
              # Var(Beta_true) = Var(g) / M
              # Var(error) = 1 - Var(g) 

x, y, beta_true = build_toy_dataset(N, M, var_g)
X = tf.constant(x, shape=[N,M], dtype=tf.float32)
Y = tf.constant(y, shape=[N,1], dtype=tf.float32)

# Could be implemented:
# building datasets using TF API without numpy


# # Precomputations - not practicable with huge datasets
# sm = np.zeros(M)
# for i in range(M):
#     sm[i] = np_squared_norm(x[:,i])


#  Parameters setup
#
# Distinction between constant and variables
# Variables: values might change between evaluation of the graph
# (if something changes within the graph, it should be a variable)

# Variables:

Emu = tf.Variable(0., dtype=tf.float32)
Ebeta = tf.Variable(tf.zeros([M,1]), dtype=tf.float32)
ny = tf.Variable(tf.zeros(M), dtype=tf.float32)
NZ = tf.Variable(0, dtype=tf.int32)
Ew = tf.Variable(0., dtype=tf.float32)
epsilon = tf.Variable(Y, dtype=tf.float32)
Sigma2_e = tf.Variable(tf_squared_norm(Y) / (N*0.5), dtype=tf.float32)
Sigma2_b = tf.Variable(rbeta(1.0,1.0), dtype=tf.float32)

# Constants:

vEmu = tf.ones([N,1])
v0E = tf.constant(0.001, dtype=tf.float32)
v0B = tf.constant(0.001, dtype=tf.float32)
s0B = Sigma2_b.initialized_value() / 2
s0E = Sigma2_e.initialized_value() / 2

# Placeholders:
Xj = tf.placeholder(tf.float32, shape=(N,1))
Bj = tf.placeholder(tf.float32, shape=())
index = tf.placeholder(tf.int32, shape=())


# Print stuff:
# TODO: construct the op with tf.print() so that values get automatically printed

print_dict = {'Emu': Emu, 'Ew': Ew, 
              'NZ': NZ, 'Sigma2_e': Sigma2_e,
              'Sigma2_b': Sigma2_b}


# Tensorboard graph

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())

# Maybe have a flow between item assignment and regular python variable
# to make things simple

# Computations
ta_beta, ta_ny, ta_eps = sample_beta(
    Xj, epsilon, Sigma2_e,
    Sigma2_b, Ew, Ebeta[index])
ta_epsilon = Y - tf.matmul(X,Ebeta) - vEmu*Emu
ta_s2b = sample_sigma2_b(Ebeta,NZ,v0B,s0B)
ta_s2e = sample_sigma2_e(N,epsilon,v0E,s0E)

# Assignment ops
emu_up = Emu.assign(sample_mu(N, Sigma2_e, Y, X, Ebeta))
beta_item_assign_op = Ebeta[index].assign(ta_beta)
ny_up = NZ.assign_add(ta_ny)
eps_up_fl = epsilon.assign(ta_eps)
eps_up = epsilon.assign(ta_epsilon)
ny_reset = NZ.assign(0)
ew_up = Ew.assign(sample_w(M,NZ))
s2b_up = Sigma2_b.assign(ta_s2b)
s2e_up = Sigma2_e.assign(ta_s2e)

up_grp = tf.group(
    beta_item_assign_op,
    ny_up,
    eps_up
)



# Number of Gibbs sampling iterations
num_iter = 30

with tf.Session() as sess:

    # Initialize variable
    sess.run(tf.global_variables_initializer())

    # Gibbs sampler iterations
    for i in range(num_iter):
        print("Gibbs sampling iteration: ", i)
        time_in = time.clock()
        sess.run(emu_up)
        sess.run(ny_reset)
        index = np.random.permutation(M)

        for marker in index:

            feed = {index:marker, Xj:X[marker]}
            sess.run(up_grp, feed_dict=feed)
        
        sess.run(emu_up)
        sess.run(eps_up)
        sess.run(s2b_up)
        sess.run(s2e_up)
        # Print operations 
        print("\n")
        print(sess.run(print_dict))
        print(" ")
        time_out = time.clock()
        print('Time for the ', i, 'th iteration: ', time_out - time_in, ' seconds')
        print(" ")
        
        
        
        # Print operations 
        print("\n")
        print(sess.run(print_dict))
        print(" ")
        time_out = time.clock()
        print('Time for the ', i, 'th iteration: ', time_out - time_in, ' seconds')
        print(" ")

