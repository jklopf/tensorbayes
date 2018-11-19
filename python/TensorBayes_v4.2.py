## Imports
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
tfd = tfp.distributions

'''
This version is able to retrieve the simulated parameters and
store the history of the sampling, and implement the tensorflow dataset API instead of
placeholders to feed data. Compared to other tensorflow-based version, the runtime has reduced of 10s.

It also implements algorithm optimization, where whole dataset matrice multiplication are avoided:
Emu is not sampled anymore.
epsilon is updated only during the dataset full pass, and not also after.

It also implements control dependencies, to minimize the number of sess.run() calls.
'''
# Start time measures
start_time = time.clock()

# Reset the graph
tf.reset_default_graph()

## Reproducibility
# Seed setting for reproducable research.

# Set numpy seed
np.random.seed(1234)

# Set graph-level seed
tf.set_random_seed(1234)


## Util functions

def tf_squared_norm(vector):
    sum_of_squares = tf.reduce_sum(tf.square(vector))
    return sum_of_squares
    
def np_squared_norm(vector):
    sum_of_squares = np.sum(np.square(vector))
    return sum_of_squares


## Distributions functions

def rnorm(mean, var):
    # rnorm is defined using the variance (i.e sigma^2)
    sd = tf.sqrt(var)
    dist = tfd.Normal(loc= mean, scale= sd)
    sample = dist.sample()
    return sample

def rbeta(a, b):
    dist = tfd.Beta(a, b)
    sample = dist.sample()
    return sample

def rinvchisq(df, scale):
	# scale factor = tau^2
    dist = tfd.Chi2(df)
    sample = (df * scale)/dist.sample()
    return sample

def rbernoulli(p):
    dist = tfd.Bernoulli(probs=p)
    sample = dist.sample()
    return sample


## Sampling functions

def sample_mu(N, Sigma2_e, Y, X, betas):
    # sample mean
    mean = tf.reduce_sum(tf.subtract(Y, tf.matmul(X, betas)))/N
    var = Sigma2_e/N
    sample = rnorm(mean,var)
    return sample

def sample_sigma2_b(betas, NZ, v0B, s0B):
    # sample variance of beta
    df = v0B+NZ
    scale = (tf_squared_norm(betas)+v0B*s0B) / df  
    sample = rinvchisq(df, scale)
    return sample

def sample_sigma2_e(N, epsilon, v0E, s0E):
    # sample error variance of Y
    df = v0E + N
    scale = (tf_squared_norm(epsilon) + v0E*s0E) / df
    sample = rinvchisq(df, scale)
    return sample

def sample_w(M, NZ):
    # sample mixture weight
    sample = rbeta(NZ+1, M-NZ+1)
    return sample

def sample_beta(x_j, eps, s2e, s2b, w, beta_old):
    # sample a beta
    eps = eps + (x_j*beta_old)
    Cj = tf_squared_norm(x_j) + s2e/s2b
    rj = tf.tensordot(tf.transpose(x_j), eps, 1)[0,0]
    ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*s2e ))) * tf.sqrt((s2b*Cj)/s2e)
    pij = w / (w + ratio*(1-w))
    toss = rbernoulli(pij)
    def case_zero():
        return 0., 0.
    def case_one():
        return rnorm(rj/Cj, s2e/Cj), 1.
    beta_new, ny_new = tf.cond(tf.equal(toss,1),case_one, case_zero)
    eps = eps - (x_j*beta_new)
    return beta_new, ny_new, eps 


## Simulate data

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
X = tf.constant(x, shape=[M,N], dtype=tf.float32) # /!\ shape is now [M,N] /!\
Y = tf.constant(y, shape=[N,1], dtype=tf.float32)

## Dataset API implementation

data_index = tf.data.Dataset.range(M) # reflect which column was selected at random
data_x = tf.data.Dataset.from_tensor_slices(X) # reflects the randomly selected column
data = tf.data.Dataset.zip((data_index, data_x)).shuffle(M) # zip together and shuffle them
iterator = data.make_initializable_iterator() # reinitializable iterator: initialize at each gibbs iteration
ind, col = iterator.get_next() # dataset element
colx = tf.reshape(col, [N,1]) # reshape the array element as a column vector


# Could be implemented:
# building datasets using TF API without numpy

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

Emu = tf.Variable(0., dtype=tf.float32)
Ebeta = tf.Variable(tf.zeros([M,1], dtype=tf.float32), dtype=tf.float32)
Ny = tf.Variable(tf.zeros(M, dtype=tf.float32), dtype=tf.float32)
NZ = tf.Variable(0., dtype=tf.float32)
Ew = tf.Variable(0., dtype=tf.float32)
epsilon = tf.Variable(Y, dtype=tf.float32)
Sigma2_e = tf.Variable(tf_squared_norm(Y) / (N*0.5), dtype=tf.float32)
Sigma2_b = tf.Variable(rbeta(1., 1.), dtype=tf.float32)

# Constants:

vEmu = tf.ones([N,1], dtype=tf.float32)
v0E = tf.constant(0.001, dtype=tf.float32)
v0B = tf.constant(0.001, dtype=tf.float32)
s0B = Sigma2_b.initialized_value() / 2
s0E = Sigma2_e.initialized_value() / 2


# Tensorboard graph
# TODO: look up what TensorBoard can do, this can be used in the end to have a graph representation of the algorithm.
# Also, for graph clarity, operations should be named.

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())


# Computations: computation for each column
ta_beta, ta_ny, ta_eps = sample_beta(colx, epsilon, Sigma2_e, Sigma2_b, Ew, Ebeta[ind,0])
# ta_epsilon = Y - tf.matmul(X,Ebeta) - vEmu*Emu | DEPRECATED


# Assignment ops:
# As we don't chain assignment operations, assignment does not require to return the evaluation of the new value
# therefore, all read_value are set to False. This changes runtime for about 1 sec (see above).
# Run with `read_value = True`: 63.4s
# Run with `read_value = False`: 62.2s

beta_item_assign_op = Ebeta[ind,0].assign(ta_beta) 		# when doing item assignment, read_value becomes an unexpected parameter, 
ny_item_assign_op = Ny[ind].assign(ta_ny)               # as tensorflow doesn't know what to return the single item or the whole variable
eps_up_fl = epsilon.assign(ta_eps, read_value=False)
fullpass = tf.group(beta_item_assign_op, ny_item_assign_op, eps_up_fl)

s2e_up = Sigma2_e.assign(sample_sigma2_e(N,epsilon,v0E,s0E), read_value=False)
nz_up = NZ.assign(tf.reduce_sum(Ny), read_value=False)
first_round = tf.group(nz_up,s2e_up)

# Control dependencies:
with tf.control_dependencies([first_round]):
	ew_up = Ew.assign(sample_w(M,NZ), read_value=False)
	s2b_up = Sigma2_b.assign(sample_sigma2_b(Ebeta,NZ,v0B,s0B), read_value=False)

param_up = tf.group(ew_up, s2b_up)
# Logs definition:
param_log = [] # order: Sigma2_e, Sigma2_b
beta_log = [] # as rank 1 vector


# Number of Gibbs sampling iterations
num_iter = 5000
burned_samples_threshold = 2000

# Launch of session
with tf.Session() as sess:

    # Initialize variable
    sess.run(tf.global_variables_initializer())

    # Gibbs sampler iterations
    print('\n', "Beginning of sampling: each dot = 250 iterations", '\n')
    for i in range(num_iter): # TODO: replace with tf.while ?

        # Print progress
        if(i%250 == 0): print(".",end='', flush=True)

            
        # While loop: dataset full pass
        sess.run(iterator.initializer)        
        while True: # Loop on 'col_next', the queue of column iterator
            try: # Run Ebeta item assign op

                sess.run(fullpass)

            except tf.errors.OutOfRangeError:

                # End of full pass, update parameters
                sess.run(param_up)

                # Exit while loop to enter next gibb iteration
                break
            
        # Logs
        if(i > burned_samples_threshold):
            
            param_log.append(sess.run([Sigma2_e, Sigma2_b]))
            beta_log.append(np.array(sess.run(Ebeta)).reshape(M))


print("\n")
print("End of sampling" + '\n')

# Time elapsed
total_time =   np.round(time.clock() - start_time, 5)
print("Time elapsed: " + str(total_time) + "s" + '\n')

# Results
param_log = np.array(param_log)
mean_Sigma2_e = np.round(np.mean(param_log[:,0]),5)
mean_Sigma2_b = np.round(np.mean(param_log[:,1]),5)
mean_betas = np.round(np.mean(beta_log, axis=0),5).reshape([M,1])

# Results printing
print("Parameters: " + '\n')
#print(" ")
print("Mean Sigma2_e:", mean_Sigma2_e,'\t', "Expected Sigma2_e:", np.round(1-var_g, 5))
print("Mean Sigma2_b:", mean_Sigma2_b,'\t', "Expected Sigma2_b:", np.round(var_g / M, 5), "\n")
print("Coefficients:" + '\n')
print("Computed" + '\t' + '\t' + "Expected" )
for i in range(M):
    print(mean_betas[i,0], '\t', '\t', beta_true[i,0] )

