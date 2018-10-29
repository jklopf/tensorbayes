
# coding: utf-8

# # TensorBayes
# 
# ### Adaptation of `BayesC.cpp`
# 
# 

# ## Imports

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time

tfd = tfp.distributions

# ## Start time measures

start_time = time.clock()

# ## Reproducibility
# 
# Seed setting for reproducable research.

#Set numpy seed
np.random.seed(1234)

# Set graph-level seed
tf.set_random_seed(1234)


# ## Distributions functions
# 


def runif(lower, higher):
    dist = tfd.Uniform(lower, higher)
    return dist.sample()

def rnorm(mean, sd):
    dist = tfd.Normal(loc= mean, scale= sd)
    return dist.sample()

def rbeta(alpha, beta):
    dist = tfd.Beta(alpha, beta)
    return dist.sample()

def rinvchisq(df, scale):
    dist = tfd.InverseGamma(df*0.5, df*scale*0.5)
    return dist.sample()

def rbernoulli(p):
    dist = tfd.Bernoulli(probs=p)
    return dist.sample()


# ## Sampling functions
# 

# sample mean
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


## Simulate data

def build_toy_dataset(N, beta, sigmaY_true=1):
    
    features = len(beta)
    x = np.random.randn(N, features)
    y = np.dot(x, beta) + np.random.normal(0, sigmaY_true, size=N)
    return x, y

N = 40  # number of data points
M = 10  # number of features

beta_true = np.random.randn(M)
x, y = build_toy_dataset(N, beta_true)

X = tf.constant(x, shape=[N,M], dtype=tf.float32)
Y = tf.constant(y, shape = [N,1], dtype=tf.float32)

index = np.random.permutation(M)

# Could be implemented:
# building datasets using TF API without numpy


# ## Parameters setup

# In[82]:


# Distinction between constant and variables
# Variables: values might change between evaluation of the graph
# (if something changes within the graph, it should be a variable)

Emu = tf.Variable(0., trainable=False)
vEmu = tf.ones([N,1])
Ebeta = np.zeros([M,1])
Ebeta_ = tf.Variable(Ebeta, dtype=tf.float32, trainable=False)
ny = np.zeros([M,1])
Ew = tf.Variable(0., trainable=False)
epsilon = tf.Variable(Y, trainable=False)
NZ = tf.Variable(0., trainable=False)
Esigma2 = tf.Variable(tf.nn.l2_loss(epsilon.initialized_value())/N, trainable=False)
Epsi2 = tf.Variable(rbeta(1.,1.), trainable=False)


# In[83]:


#Standard parameterization of hyperpriors for variances
#double v0E=0.001,s0E=0.001,v0B=0.001,s0B=0.001;

#Alternative parameterization of hyperpriors for variances
v0E, v0B = 4, 4
s0B=((v0B-2)/v0B)*Epsi2
s0E=((v0E-2)/v0E)*Esigma2


# ## Tensorboard graph

# In[84]:


writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())


# ## Gibbs sampling

# In[85]:


# Open session
sess = tf.Session()


# In[86]:


# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)


num_iter = 50

update = tf.group(
	Ebeta_.assign(Ebeta),
	epsilon.assign(Y-tf.matmul(X,Ebeta_)-vEmu*Emu),
	Emu.assign(sample_mu(N, Esigma2, Y, X, Ebeta_)),
	NZ.assign(np.sum(ny)),
	Ew.assign(sample_w(M,NZ)),
	Epsi2.assign(sample_psi2_chisq(Ebeta_,NZ,v0B,s0B)),
	Esigma2.assign(sample_sigma_chisq(N,epsilon,v0E,s0E)))




# In[ ]:


for i in range(num_iter):
	print("Gibbs sampling iteration: ", i)
	#sess.run(u_Emu)
	for marker in index:
		sess.run(epsilon.assign_add(tf.reshape(X[:,marker]*Ebeta[marker],[N,1])))
		Cj=tf.nn.l2_loss(X[:,marker])*2+Esigma2/Epsi2 #adjusted variance
		rj= tf.matmul(tf.reshape(X[:,marker], [1,N]),tf.reshape(epsilon, [N,1])) # mean
		ratio=((tf.exp(-(tf.pow(rj,2))/(2*Cj*Esigma2))*tf.sqrt((Epsi2*Cj)/Esigma2)))
		ratio=Ew/(Ew+ratio*(1-Ew))

		ny[marker] = sess.run(rbernoulli(ratio))

		if (ny[marker]==0):
			Ebeta[marker]=0

		elif (ny[marker]==1):
			Ebeta[marker]=sess.run(rnorm(rj/Cj,Esigma2/Cj))

		sess.run(epsilon.assign_sub(tf.reshape(X[:,marker]*Ebeta[marker],[N,1])))

	#for i in range(len(Ebeta)):
	#    print(Ebeta[i], "\t", ny[i])
	# sess.run(u_Ebeta_)
	# sess.run(u_NZ)
	# sess.run(u_Ew)
	# sess.run(u_epsilon)
	# sess.run(u_epsi2)
	# sess.run(u_Esigma2)
	#print(sess.run(Ebeta))
	sess.run(update)

# ## End session
sess.close()

# ## Print results
print("Ebeta" + "\t" + ' ny' + '\t'+ ' beta_true')
for i in range(M):
    print(Ebeta[i], "\t", ny[i], "\t", beta_true[i])


# ## Printe time
print('Time elapsed: ')
print(time.clock() - start_time, "seconds")
