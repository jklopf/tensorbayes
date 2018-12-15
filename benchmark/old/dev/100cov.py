## Imports
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from timeit import repeat
tfd = tfp.distributions
import psutil
from tqdm import tqdm

'''
This version is able to retrieve the simulated parameters and
store the history of the sampling, and implement the tensorflow dataset API instead of
placeholders to feed data. Compared to other tensorflow-based version, the runtime has reduced of 10s.

It also implements algorithm optimization, where whole dataset matrice multiplication are avoided:
Emu is not sampled anymore.
epsilon is updated only during the dataset full pass, and not also after.

It also implements control dependencies, to minimize the number of sess.run() calls.
'''

# Reset the graph
tf.reset_default_graph()

## Reproducibility
# Seed setting for reproducable research.

# Set numpy seed
np.random.seed(1234)

# Print options (dev)
# np.set_printoptions(precision=5, floatmode='fixed')

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
    return x, y, beta_true

# Simulated data parameters
'''
Var(b) = Var(g) / M
Var(e) = 1 - Var(g)
'''

N = 500       # number of data points
M = 100        # number of features
var_g = 0.7   # genetic variance parameter

# Benchmark parameters and logs
# oa: overall
n_time = 1
oa_mean_s2b = []
oa_mean_s2e = []
oa_cor = []
oa_pip = []



def gibb():
    global N, M
    global oa_mean_s2b
    global oa_mean_s2e
    global oa_cor, oa_pip

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

    Ebeta = tf.Variable(tf.zeros([M,1], dtype=tf.float32), dtype=tf.float32)
    Ny = tf.Variable(tf.zeros(M, dtype=tf.float32), dtype=tf.float32)
    NZ = tf.Variable(0., dtype=tf.float32)
    Ew = tf.Variable(0., dtype=tf.float32)
    epsilon = tf.Variable(Y, dtype=tf.float32)
    Sigma2_e = tf.Variable(tf_squared_norm(Y) / (N*0.5), dtype=tf.float32)
    Sigma2_b = tf.Variable(rbeta(1., 1.), dtype=tf.float32)

    # Constants:

    v0E = tf.constant(-2.0, dtype=tf.float32)
    v0B = tf.constant(-2.0, dtype=tf.float32)
    s0B = Sigma2_b.initialized_value() / 2
    s0E = Sigma2_e.initialized_value() / 2


    # Tensorboard graph
    # TODO: look up what TensorBoard can do, this can be used in the end to have a graph representation of the algorithm.
    # Also, for graph clarity, operations should be named.

    #writer = tf.summary.FileWriter('.')
    #writer.add_graph(tf.get_default_graph())


    # Computations: computation for each column
    # ta: to assign
    ta_beta, ta_ny, ta_eps = sample_beta(colx, epsilon, Sigma2_e, Sigma2_b, Ew, Ebeta[ind,0])


    # Assignment ops:
    # As we don't chain assignment operations, assignment does not require to return the evaluation of the new value
    # therefore, all read_value are set to False. This changes runtime for about 1 sec (see above).
    # Run with `read_value = True`: 63.4s
    # Run with `read_value = False`: 62.2s
    # maybe there is a trick here for storing the log using read_value

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
    ny_log = []


    # Number of Gibbs sampling iterations
    num_iter = 5000
    burned_samples_threshold = 2000

    # Launch of session
    with tf.Session() as sess:

        # Initialize variable
        sess.run(tf.global_variables_initializer())

        # Gibbs sampler iterations
        for i in tqdm(range(num_iter)): # TODO: replace with tf.while ?

            # While loop: dataset full pass
            sess.run(iterator.initializer)        
            while True: # Loop on 'col_next', the queue of column iterator
                try: # Run Ebeta item assign op

                    sess.run(fullpass)

                except tf.errors.OutOfRangeError:

                    # End of full pass, update parameters
                    sess.run(param_up)

                    # Exit while loop to enter next Gibbs iteration
                    break
                
            # Store sampling logs
            if(i >= burned_samples_threshold):
                
                param_log.append(sess.run([Sigma2_e, Sigma2_b]))
                beta_log.append(np.array(sess.run(Ebeta)).reshape(M))
                ny_log.append(sess.run(Ny))
    

    # Store local results
    param_log = np.array(param_log) # [s2e, s2b]
    mean_s2e = np.mean(param_log[:,0])
    mean_s2b = np.mean(param_log[:,1])
    mean_ebeta = np.mean(beta_log, axis=0)
    pip = np.mean(ny_log, axis = 0)
    corr_ebeta_betatrue = np.corrcoef(mean_ebeta, beta_true)[0][1]

    # Store overall results
    oa_mean_s2e.append(mean_s2e)
    oa_mean_s2b.append(mean_s2b)
    oa_pip.append(len([num for num in pip if num >= 0.95]))
    oa_cor.append(corr_ebeta_betatrue)



# Measure running times and execute the code n_time
oa_time = repeat('gibb()',repeat=n_time, number=1, setup='from __main__ import gibb')


# Measure memory usage
mem = psutil.Process().memory_info()
rss = np.round(mem.rss / (1024**2), 5)
vms = np.round(mem.vms / (1024**2), 5)

# Output benchmark logs
print('\nD1 Logs: TensorBayes v4.2 on CPU')
print('N = {}, M = {}, var(g) = {}'.format(N,M,var_g))
print('\nrss memory (physical): {} MiB'.format(rss))
print('vms memory (virtual): {} MiB'.format(vms))
print('\nMin time of execution: ', np.round(np.min(oa_time), 4))
print('Mean time of execution memory: ', np.round(np.mean(oa_time),4))

# Write results to a .csv
# Order: s2e | s2b | cor | pip | time
results = np.stack((
    oa_mean_s2e,
    oa_mean_s2b,
    oa_cor,
    oa_pip,
    oa_time), axis=-1)

np.set_printoptions(
   formatter={'float_kind':'{:0.5f}'.format})
print('s2e | s2b | cor | pip | time')
print(results)

#np.savetxt(
#    'dev.csv',
#    results,
#    delimiter=',',
#    header='sigma2_e, sigma2_b, cor(eb,bt), PiP, time',
#    fmt='%.8f')