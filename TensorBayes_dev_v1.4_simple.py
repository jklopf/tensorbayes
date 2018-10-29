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
    return dist.sample()

def rbeta(alpha, beta):
    dist = tfd.Beta(alpha, beta)
    return dist.sample()

def rinvchisq(df, scale): # scale factor = tau^2
    dist = tfd.InverseGamma(df*0.5, df*scale*0.5)
    return dist.sample()

def rbernoulli(p):
    dist = tfd.Bernoulli(probs=p)
    return dist.sample()


# Sampling functions
# 

# sample mean
def sample_mu(N, Sigma2_e, Y, X, beta):    
    mean = tf.reduce_sum(tf.subtract(Y, tf.matmul(X, beta)))/N
    var = Sigma2_e/N    
    return rnorm(mean, var)

# sample variance of beta
def sample_sigma2_b( beta, NZ, v0B, s0B):
    df = v0B+NZ
    scale = (tf_squared_norm(beta)*NZ+v0B*s0B) / df  
    return rinvchisq(df, scale)


# sample error variance of Y
def sample_sigma2_e( N, epsilon, v0E, s0E):
    df = v0E + N
    scale = (tf_squared_norm(epsilon) + v0E*s0E) / df
    return rinvchisq(df, scale)

# sample mixture weight
def sample_w( M, NZ):
    w=rbeta(1+NZ,1+M-NZ)
    return w

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


# x, y, beta_true = build_toy_dataset(N, M, var_g)

# X = tf.constant(x, shape=[N,M], dtype=tf.float32)
# Y = tf.constant(y, shape = [N,1], dtype=tf.float32)

                # Could be implemented:
                # building datasets using TF API without numpy

# Alternative simulated data

beta_true = tf.constant(0.25, shape=[M,1], dtype=tf.float32)
x = np.random.randn(N,M)
X = tf.constant(x, dtype = tf.float32)
Y = tf.matmul(X, beta_true) + (tf.random_normal([N,1], dtype=tf.float32) * 0.375)

 # Precomputations
sm = np.zeros(M)
for i in range(M):
    sm[i] = np_squared_norm(x[:,i])
    


#  Parameters setup
#
# Distinction between constant and variables
# Variables: values might change between evaluation of the graph
# (if something changes within the graph, it should be a variable)

Emu = tf.Variable(0., dtype=tf.float32 , trainable=False)
vEmu = tf.ones([N,1])
Ebeta = np.zeros((M,1), dtype = "float32")
#Ebeta_ = tf.Variable(Ebeta, dtype=tf.float32, trainable=False)
Ebeta_ = tf.placeholder(tf.float32, shape=(M,1))
ny = np.zeros([M,1])
Ew = tf.Variable(0., trainable=False)
epsilon = tf.Variable(Y, trainable=False)
NZ = tf.Variable(0., trainable=False)
Sigma2_e = tf.Variable(tf_squared_norm(Y) / (N*0.5), trainable=False)
Sigma2_b = tf.Variable(rbeta(1.0,1.0), trainable=False)

colx = tf.placeholder(tf.float32, shape=(N,1))
ind_ = tf.placeholder(tf.int32, shape=())


# Alternatives parameterization of hyperpriors for variances
v0E = tf.constant(4.0)
v0B = tf.constant(4.0)
s0B = Sigma2_b.initialized_value() / 2
s0E = Sigma2_e.initialized_value() / 2


print_dict = {'Emu': Emu, 'Ew': Ew, 
              'NZ': NZ, 'Sigma2_e': Sigma2_e,
              'Sigma2_b': Sigma2_b}

# Tensorboard graph

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())

# updates ops
#Emu_up = Emu.assign(sample_mu(N, Sigma2_e, Y, X, Ebeta_))

#eps_plus = epsilon.assign_add(colx * Ebeta[ind_])
#eps_minus = epsilon.assign_sub(colx * Ebeta[ind_])



#sess.run(Cj.assign(tf.reduce_sum(tf.pow(X[:,marker],2)) + Sigma2_e/Sigma2_b)) #adjusted variance
#sess.run(rj.assign(tf.matmul(tf.reshape(X[:,marker], [1,N]),epsilon)[0][0])) # mean, tensordot instead of matmul ?
#sess.run(ratio.assign(tf.exp(-(tf.pow(rj,2))/(2*Cj*Sigma2_e))*tf.sqrt((Sigma2_b*Cj)/Sigma2_e)))
#sess.run(pij.assign(Ew/(Ew+ratio*(1-Ew))))

def cond_true():
    
    return rnorm(rj/Cj,Sigma2_e/Cj)

def cond_false():
    
    return 0.
    


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
        Emu = sample_mu(N, Sigma2_e, Y, X, Ebeta)
        
        # Reset NZ parameter
        sess.run(NZ.assign(0.))
        
        # Compute beta for each marker
        #print("Current marker:", end=" ")
        print("Current marker:")
        for marker in index:
            print(marker, end=" ", flush=True)

            epsilon += x[:,marker].reshape(N,1) * Ebeta[marker]  
            Cj = sm[marker] + Sigma2_e/Sigma2_b
            rj = tf.tensordot(tf.transpose(colx), epsilon, 1)[0]
            ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*Sigma2_e ))) * tf.sqrt((Sigma2_b*Cj)/Sigma2_e)
            pij = Ew / (Ew + ratio*(1-Ew))
            #ny[marker] = sess.run(rbernoulli(pij), feed_dict={colx: x[:,marker].reshape(N,1)})

            # TODO: replace with tf.cond | DONE
            
            Ebeta[marker] = sess.run(tf.cond(tf.equal(rbernoulli(pij)[0],1),cond_true, cond_false),
                 feed_dict={colx: x[:,marker].reshape(N,1)})
            
            sess.run(tf.cond(Ebeta[marker][0] != 0.,lambda: NZ.assign_add(1.), lambda: NZ.assign_add(0.)))
            
            
            
            #if (ny[marker]==0):
            #    Ebeta[marker]=0

            #elif (ny[marker]==1):
            #    Ebeta[marker] = sess.run(rnorm(rj/Cj,Sigma2_e/Cj), feed_dict={colx: x[:,marker].reshape(N,1)})

            epsilon -= x[:,marker].reshape(N,1) * Ebeta[marker]
            

        #for i in range(len(Ebeta)):
        #    print(Ebeta[i], "\t", ny[i])
        #sess.run(NZ.assign(np.sum(ny)))
        Ew = sample_w(M,NZ)
        #sess.run(Ebeta_.assign(Ebeta))
        epsilon = Y - tf.matmul(X,Ebeta) - vEmu*Emu             
        Sigma2_b = sample_sigma2_b(Ebeta,NZ,v0B,s0B)
                 
        Sigma2_e = sample_sigma2_e(N,epsilon,v0E,s0E)
        
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



