
# coding: utf-8

# # TensorBayes
# 
# ### Adaptation of `BayesC.cpp`
# 
# 

# ## Imports

# In[1]:


import tensorflow as tf


# In[3]:


import tensorflow_probability as tfp


# In[2]:


import numpy as np


# In[4]:


tfd = tfp.distributions


# ## File input
# 
# To do

# In[2]:


# Get the numbers of columns in the csv:
# File I/O here 
filenames = ""

csv_in = open(filenames, "r")                        # open the csv
ncol = len(csv_in.readline().split(","))            # read the first line and count the # of columns
csv_in.close()                                      # close the csv
pr("Number of columns in the csv: " + str(ncol)) # pr the # of columns


# ## Reproducibility
# 
# Seed setting for reproducable research.

# In[ ]:


# To do: get a numpy seed or look at how TF implements rng.

# each distributions.sample() seen below can be seedeed.
# ex. dist.sample(seed=32): return a sample of shape=() (scalar).


# ## Distributions functions
# 
# - Random Uniform:   
# return a sample from a uniform distribution of limits parameter `lower ` and `higher`.
#    
#    
# - Random Normal:   
# return a sample from a normal distribution of parameter `mean` and `standard deviation`.
#    
#    
# - Random Beta:   
# return a random quantile of a beta distribution of parameter `alpha` and `beta`.
#    
#    
# - Random Inversed Chi$^2$:   
# return a random quantile of a inversed chi$^2$ distribution of parameter `degrees of freedom` and `scale`.
#    
#    
# - Random Bernoulli:   
# return a sample from a bernoulli distribution of probability of sucess `p`.
#    
#    

# In[5]:


# Note: written as a translation of BayesC.cpp
# the function definitions might not be needeed,
# and the declarations of the distributions could be enough

def runif(lower, higher):
    dist = tfd.Uniform(lower, higher)
    return dist.sample()

def rnorm(mean, sd):
    dist = tfd.Normal(loc= mean, scale= sd)
    return dist.sample()

def rbeta(alpha, beta):
    dist = tfd.Beta(alpha, beta)
    q = dist.quantile(runif(0,1)) # note: runif(0,1) could be declared to avoid add ops
    return q

def rinvchisq(df, scale):
    dist = tfd.InverseGamma(df*0.5, df*scale*0.5)
    q = dist.quantile(runif(0,1))
    return q

def rbernoulli(p):
    dist = tfd.Bernoulli(probs=p)
    return dist.sample()



def sample_mu(N, Esigma2, Y, X, beta): #as in BayesC, with the N parameter
    mean = tf.reduce_sum(tf.subtract(Y, tf.matmul(X, beta)))/N
    sd = tf.sqrt(Esigma2/N)
    mu = rnorm(mean, sd)
    return mu

# sample variance of beta
 def sample_psi2_chisq( beta, NZ, v0B, s0B):
	 df=v0B+NZ
	 scale=(tf.nn.l2_loss(beta)*2*NZ+v0B*s0B)/(v0B+NZ)
	 psi2=rinvchisq(df, scale)
	return(psi2)


# sample error variance of Y
 def sample_sigma_chisq( N, epsilon, v0E, s0E):
	 sigma2=rinvchisq(v0E+N, (tf.nn.l2_loss(epsilon)*2+v0E*s0E)/(v0E+N))
	return(sigma2)


# sample mixture weight
 def sample_w( M, NZ):
	 w=rbeta(1+NZ,1+(M-NZ))
	return(w)


def build_toy_dataset(N, beta, noise_std=0.1):
    
    D = len(beta)
    x = np.random.randn(N, M)
    y = np.dot(x, beta) + np.random.normal(0, noise_std, size=N)
    return x, y

N = 40  # number of data points
M = 10  # number of features

beta_true = np.random.randn(M)
X, Y = build_toy_dataset(N, beta_true)
    


Emu = tf.constant([0])
vEmu = tf.ones(N)
Ebeta = tf.zeros(M)
ny = tf.zeros(M)
Ew = tf.constant([0.5])
epsilon = Y - X*Ebeta - vEmu*Emu
NZ = tf.constant([0])

Esigma2 = tf.nn.l2_loss(epsilon)/N
Epsi2 = rbeta(1,1)

#Standard parameterization of hyperpriors for variances
#double v0E=0.001,s0E=0.001,v0B=0.001,s0B=0.001;

#Alternative parameterization of hyperpriors for variances
v0E=4,v0B=4
s0B=((v0B-2)/v0B)*Epsi2
s0E=((v0E-2)/v0E)*Esigma2

# pre-computed elements for calculations
el1 = []
for i in M:
	el1[i] = X[:,i] * np.transpose(X[:,i])




# 

# Create random column order list (dataset) + iterator
col_list = tf.data.Dataset.range(ncol).shuffle(buffer_size=ncol)
col_next = col_list.make_one_shot_iterator().get_next()

#def scale_zscore(vector):
#    mean, var = tf.nn.moments(vector, axes=[0])
#    normalized_col = tf.map_fn(lambda x: (x - mean)/tf.sqrt(var), vector)
#    return normalized_col

# Launch of graph
with tf.Session() as sess:

    while True: # Loop on 'col_next', the queue of column iterator
        try:
            index = sess.run(col_next)
            dataset = tf.contrib.data.CsvDataset( # Creates a dataset of the current csv column
                        "ex.csv",
                        [tf.float32],
                        select_cols=[index]  # Only parse last three columns
                    )
            next_element = dataset.make_one_shot_iterator().get_next() # Creates an iterator
            pr('Current column to be full pass: ' + str(index))
            current_col = []
            while True: 
                try:
                    current_col.append(sess.run(next_element)[0]) # Full pass
                except tf.errors.OutOfRangeError: # End of full pass
                    
                    pr(current_col)
                    current_col = tf.convert_to_tensor([current_col])
                    mean, var = tf.nn.moments(current_col, axes=[0])
                    normalized_col = tf.map_fn(lambda x: (x - mean)/tf.sqrt(var), current_col)
                    pr(normalized_col)
                    pr('\n')
                    
                    break


            

        except tf.errors.OutOfRangeError:
            break



