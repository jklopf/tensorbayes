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

## Simulate data

# Var(g) = 0.7
# Var(Beta_true) = Var(g) / M
# Var(error) = 1 - Var(g) 


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
NZ = tf.Variable(0., dtype=tf.float32)
Ew = tf.Variable(0., dtype=tf.float32)
epsilon = tf.Variable(Y, dtype=tf.float32)
Sigma2_e = tf.Variable(tf_squared_norm(Y) / (N*0.5), dtype=tf.float32)
Sigma2_b = tf.Variable(rbeta(1.0,1.0), dtype=tf.float32)

# Constants:

vEmu = tf.constant(tf.ones([N,1]))
v0E = tf.constant(0.001, dtype=tf.float32)
v0B = tf.constant(0.001, dtype=tf.float32)
s0B = tf.constant(Sigma2_b.initialized_value()/2, dtype=tf.float32)
s0E = tf.constant(Sigma2_e.initialized_value()/2, dtype=tf.float32)

# Placeholders:
Xj = tf.placeholder(tf.float32, shape=(N,1))


# Print stuff:
#TODO: construct the op with tf.print() so that values get automatically printed

print_dict = {'Emu': Emu, 'Ew': Ew, 
              'NZ': NZ, 'Sigma2_e': Sigma2_e,
              'Sigma2_b': Sigma2_b}


# Tensorboard graph

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())

# updates ops
# Emu_up = Emu.assign(sample_mu(N, Sigma2_e, Y, X, Ebeta))




#eps_plus = epsilon.assign_add(colx * Ebeta[ind_])
#eps_minus = epsilon.assign_sub(colx * Ebeta[ind_])
#sess.run(Cj.assign(tf.reduce_sum(tf.pow(X[:,marker],2)) + Sigma2_e/Sigma2_b)) #adjusted variance
#sess.run(rj.assign(tf.matmul(tf.reshape(X[:,marker], [1,N]),epsilon)[0][0])) # mean, tensordot instead of matmul ?
#sess.run(ratio.assign(tf.exp(-(tf.pow(rj,2))/(2*Cj*Sigma2_e))*tf.sqrt((Sigma2_b*Cj)/Sigma2_e)))
#sess.run(pij.assign(Ew/(Ew+ratio*(1-Ew))))

def for_loop(M,N, epsilon, x, Ebeta, sigma2_e, sigma2_b, Ew, ny):
    index = np.random.permutation(M)
    for marker in index:

        colx = tf.reshape(x[:,marker], [N,1])
        eps = epsilon + colx * Ebeta[marker]
        Cj = tf_squared_norm(colx) + sigma2_e/sigma2_b
        rj = tf.tensordot(tf.transpose(colx), eps, 1)[0]
        ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*Sigma2_e ))) * tf.sqrt((Sigma2_b*Cj)/Sigma2_e)
        pij = Ew / (Ew + ratio*(1-Ew))
        toss = rbernoulli(pij)
        def case_zero(): return 0.
        def case_one(): return rnorm(rj/Cj, sigma2_e/Cj)
        #TODO: PROBLEM HERE 
        # ebeta is a tensor and not a variable in this function
        # and cannot undergo item assignment that way
        Ebeta[marker] = tf.case([tf.equal(toss, 0), case_zero], default=case_one)
        ny[marker] = toss
        epsilon -= colx * Ebeta[marker]
    
    return Ebeta, ny




# assignment ops
Emu_up = Emu.assign(sample_mu(N, Sigma2_e, Y, X, Ebeta))

with tf.control_dependencies([Emu_up]):
    
    ta_beta, ta_ny = for_loop(M,N, epsilon, x, Ebeta, Sigma2_e, Sigma2_b, Ew, ny)
    ebeta_up = Ebeta.assign(ta_beta)
    nz_up = NZ.assign(ta_ny)
    with tf.control_dependencies([ebeta_up, nz_up]):
        eps_up = epsilon.assign(Y-tf.matmul(X,Ebeta)-vEmu*Emu)
        s2b_up = Sigma2_b.assign(sample_sigma2_b(Ebeta_,NZ,v0B,s0B))
    

        sess.run(NZ.assign(np.sum(ny)))
        ew_up = Ew.assign(sample_w(M,NZ)))
        #sess.run(Ebeta_.assign(Ebeta))
        sess.run(epsilon.assign(Y-tf.matmul(X,Ebeta_)-vEmu*Emu), feed_dict= {Ebeta_: Ebeta})                 
        , feed_dict= {Ebeta_: Ebeta})
                 
        sess.run(Sigma2_e.assign(sample_sigma2_e(N,epsilon,v0E,s0E)))
        


        

for marker in index:

            sess.run(epsilon.assign_add(colx * Ebeta[marker]), feed_dict={colx: x[:,marker].reshape(N,1)})          
            Cj = sm[marker] + Sigma2_e/Sigma2_b
            rj = tf.tensordot(tf.transpose(colx), epsilon, 1)[0]
            ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*Sigma2_e ))) * tf.sqrt((Sigma2_b*Cj)/Sigma2_e)
            pij = Ew / (Ew + ratio*(1-Ew))
            ny[marker] = sess.run(rbernoulli(pij), feed_dict={colx: x[:,marker].reshape(N,1)})

            # TODO: replace with tf.cond 
            if (ny[marker]==0):
                Ebeta[marker]=0

            elif (ny[marker]==1):
                Ebeta[marker] = sess.run(rnorm(rj/Cj,Sigma2_e/Cj), feed_dict={colx: x[:,marker].reshape(N,1)})

            sess.run(epsilon.assign_sub(colx * Ebeta[marker]), feed_dict={colx: x[:,marker].reshape(N,1)})  
            


# Number of Gibbs sampling iterations
num_iter = 30

with tf.Session() as sess:
    
    # Initialize variable
    sess.run(tf.global_variables_initializer())

    # Begin Gibbs iterations
    for i in range(num_iter):
        
        time_in = time.clock()
        # Print progress
        print("Gibbs sampling iteration: ", i)
        
        # Assign a random order of marker
        index = np.random.permutation(M)
        
        # Sample mu
        sess.run(Emu.assign(sample_mu(N, Sigma2_e, Y, X, Ebeta_)),
                 feed_dict={Ebeta_: Ebeta}) # matmul here
        
        # Compute beta for each marker
        #print("Current marker:", end=" ")
        print("Current marker:")
        for marker in index:
            print(marker, end=" ", flush=True)

            sess.run(epsilon.assign_add(colx * Ebeta[marker]), feed_dict={colx: x[:,marker].reshape(N,1)})          
            Cj = sm[marker] + Sigma2_e/Sigma2_b
            rj = tf.tensordot(tf.transpose(colx), epsilon, 1)[0]
            ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*Sigma2_e ))) * tf.sqrt((Sigma2_b*Cj)/Sigma2_e)
            pij = Ew / (Ew + ratio*(1-Ew))
            ny[marker] = sess.run(rbernoulli(pij), feed_dict={colx: x[:,marker].reshape(N,1)})

            # TODO: replace with tf.cond 
            if (ny[marker]==0):
                Ebeta[marker]=0

            elif (ny[marker]==1):
                Ebeta[marker] = sess.run(rnorm(rj/Cj,Sigma2_e/Cj), feed_dict={colx: x[:,marker].reshape(N,1)})

            sess.run(epsilon.assign_sub(colx * Ebeta[marker]), feed_dict={colx: x[:,marker].reshape(N,1)})  
            

        #for i in range(len(Ebeta)):
        #    print(Ebeta[i], "\t", ny[i])
        sess.run(NZ.assign(np.sum(ny)))
        sess.run(Ew.assign(sample_w(M,NZ)))
        #sess.run(Ebeta_.assign(Ebeta))
        sess.run(epsilon.assign(Y-tf.matmul(X,Ebeta_)-vEmu*Emu), feed_dict= {Ebeta_: Ebeta})                 
        sess.run(Sigma2_b.assign(sample_sigma2_b(Ebeta_,NZ,v0B,s0B)), feed_dict= {Ebeta_: Ebeta})
                 
        sess.run(Sigma2_e.assign(sample_sigma2_e(N,epsilon,v0E,s0E)))
        
        # Print operations 
        print("\n")
        print(sess.run(print_dict))
        print(" ")
        time_out = time.clock()
        print('Time for the ', i, 'th iteration: ', time_out - time_in, ' seconds')
        print(" ")




# ## Print results
print("Ebeta" + "\t" + ' ny' + '\t'+ ' beta_true')
for i in range(M):
    print(Ebeta[i], "\t", ny[i], "\t", beta_true[i])


# ## Printe time
print('Time elapsed: ')
print(time.clock() - start_time, "seconds")



