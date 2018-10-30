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

def squared_norm(vector):
    sum_of_squares = tf.reduce_sum(tf.square(vector))
    return sum_of_squares
    

# ## Distributions functions


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
    mu = rnorm(mean, var)
    return mu

# sample variance of beta
def sample_sigma2_b( beta, NZ, v0B, s0B):
    df=v0B+NZ
    scale=(squared_norm(beta)*NZ+v0B*s0B) / (v0B+NZ)
    psi2=rinvchisq(df, scale)
    return psi2


# sample error variance of Y
def sample_sigma2_e( N, epsilon, v0E, s0E):
    sigma2=rinvchisq(v0E+N, (tf.nn.l2_loss(epsilon)*2+v0E*s0E)/(v0E+N))
    return(sigma2)


# sample mixture weight
def sample_w( M, NZ):
    w=rbeta(1+NZ,1+(M-NZ))
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
X = tf.random_normal([N,M])
Y = tf.matmul(X, beta_true) + (tf.random_normal([N,1]) * 0.375)



#  Parameters setup
#
# Distinction between constant and variables
# Variables: values might change between evaluation of the graph
# (if something changes within the graph, it should be a variable)

Emu = tf.Variable(0., dtype=tf.float32 , trainable=False)
vEmu = tf.ones([N,1])
Ebeta = np.zeros([M,1])
Ebeta_ = tf.Variable(Ebeta, dtype=tf.float32, trainable=False)
ny = np.zeros([M,1])
Ew = tf.Variable(0., trainable=False)
epsilon = tf.Variable(Y, trainable=False)
NZ = tf.Variable(0., trainable=False)
Sigma2_e = tf.Variable(squared_norm(Y) / (N*0.5), trainable=False)
Sigma2_b = tf.Variable(rbeta(1.0,1.0), trainable=False)
#Cj = tf.Variable(0.0, dtype=tf.float32, trainable=False)
#rj = tf.Variable(0.0, dtype=tf.float32, trainable=False)
#ratio = tf.Variable(0.0, dtype=tf.float32, trainable=False)
#pij = tf.Variable(0.0, dtype=tf.float32, trainable=False)

# Standard parameterization of hyperpriors for variances
# v0E=tf.constant(0.001)
# s0E=tf.constant(0.001)
# v0B=tf.constant(0.001)
# s0B=tf.constant(0.001)

# Alternatives parameterization of hyperpriors for variances
v0E = tf.constant(4.0)
v0B = tf.constant(4.0)
#s0B = tf.get_variable("s0B", initializer= Sigma2_b.initialized_value() * 0.5)
#s0E = tf.get_variable("s0E", initializer= Sigma2_e.initialized_value() * 0.5)
s0B = Sigma2_b.initialized_value() / 2
s0E = Sigma2_e.initialized_value() / 2


print_dict = {'Emu': Emu, 'Ew': Ew, 
              'NZ': NZ, 'Sigma2_e': Sigma2_e,
              'Sigma2_b': Sigma2_b}

# Tensorboard graph

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())

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
        sess.run(Emu.assign(sample_mu(N, Sigma2_e, Y, X, Ebeta_))) # matmul here
        
        # Compute beta for each marker
        #print("Current marker:", end=" ")
        print("Current marker:")
        for marker in index:
            print(marker, end=" ", flush=True)

            sess.run(epsilon.assign_add(tf.reshape(X[:,marker]*Ebeta[marker],[N,1])))
            #sess.run(Cj.assign(tf.reduce_sum(tf.pow(X[:,marker],2)) + Sigma2_e/Sigma2_b)) #adjusted variance
            #sess.run(rj.assign(tf.matmul(tf.reshape(X[:,marker], [1,N]),epsilon)[0][0])) # mean, tensordot instead of matmul ?
            #sess.run(ratio.assign(tf.exp(-(tf.pow(rj,2))/(2*Cj*Sigma2_e))*tf.sqrt((Sigma2_b*Cj)/Sigma2_e)))
            #sess.run(pij.assign(Ew/(Ew+ratio*(1-Ew))))
            Cj = squared_norm(X[:,marker]) + Sigma2_e/Sigma2_b
            rj = tf.tensordot(X[:,marker], epsilon, 1)[0]
            ratio = tf.exp( - ( tf.square(rj) / ( 2*Cj*Sigma2_e ))) * tf.sqrt((Sigma2_b*Cj)/Sigma2_e)
            pij = Ew / (Ew + ratio*(1-Ew))
            ny[marker] = sess.run(rbernoulli(pij))


            if (ny[marker]==0):
                Ebeta[marker]=0

            elif (ny[marker]==1):
                Ebeta[marker]=sess.run(rnorm(rj/Cj,Sigma2_e/Cj))

            sess.run(epsilon.assign_sub(tf.reshape(X[:,marker]*Ebeta[marker],[N,1])))
            

        #for i in range(len(Ebeta)):
        #    print(Ebeta[i], "\t", ny[i])
        sess.run(NZ.assign(np.sum(ny)))
        sess.run(Ew.assign(sample_w(M,NZ)))
        sess.run(Ebeta_.assign(Ebeta))
        sess.run(epsilon.assign(Y-tf.matmul(X,Ebeta_)-vEmu*Emu))
        sess.run(Sigma2_b.assign(sample_sigma2_b(Ebeta_,NZ,v0B,s0B)))
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


# ## Print time
print('Time elapsed: ')
print(time.clock() - start_time, "seconds")
