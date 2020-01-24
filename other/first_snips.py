def sample_sigma2_b(betas, NZ, v0B, s0B):
    # sample variance of betas
    df = v0B+NZ
    scale = (tf_squared_norm(betas)+v0B*s0B)/df 
    sample = rinvchisq(df, scale)
return sample
def sample_sigma2_e(N, epsilon, v0E, s0E):
    # sample variance of residuals
    df = v0E + N
    scale = (tf_squared_norm(epsilon)+v0E*s0E)/df
    sample = rinvchisq(df, scale)
return sample
def sample_pi(M, mNZ):
    # sample mixture weight
    sample = rbeta(mNZ+1, M-mNZ+1)
    return sample

# Variables
Ebeta = np.zeros((M,1)) #Numpy array
delta = np.zeros([M,1]) #Numpy array
NZ = tf.Variable(0.)
epsilon = tf.Variable(Y)
Pi = tf.Variable(0.5)
Sigma2_e = tf.Variable(tf_squared_norm(Y) / (N*0.5))
Sigma2_b = tf.Variable(rbeta(1.0,1.0))
# Placeholders
Ebeta_ = tf.placeholder(tf.float32, shape=(M,1))
colx = tf.placeholder(tf.float32, shape=(N,1))
ind_ = tf.placeholder(tf.int32, shape=())
# Constants
# Parameterization of hyperpriors for variances
v0E = tf.constant(4.0)
v0B = tf.constant(4.0)
s0B = Sigma2_b.initialized_value() / 2
s0E = Sigma2_e.initialized_value() / 2

with tf.Session() as sess:
    # Initialize variable
    sess.run(tf.global_variables_initializer())
    # Iterate Gibbs sampling
    # scheme 'num_iter' times
    for i in range(num_iter):
        # Set a new random order of marker
        index = np.random.permutation(M)
        # Parse and process columns in random order
        for marker in index:
            sess.run(epsilon.assign_add(colx * Ebeta[marker]),
                    feed_dict={colx: x[:,marker].reshape(N,1)})          
            Cj = tf_squared_norm(colx) + Sigma2_e/Sigma2_b
            rj = tf.tensordot(tf.transpose(colx), epsilon, 1)[0]
            ratio = tf.exp(-(tf.square(rj)/(2*Cj*Sigma2_e)))\
                        *tf.sqrt((Sigma2_b*Cj)/Sigma2_e)
            pij = Pi / (Pi + ratio*(1-Pi))
            delta[marker] = sess.run(rbernoulli(pij),\
                                feed_dict={colx: x[:,marker].reshape(N,1)})
            # Beta(j) conditionnal on delta(j)
            if (delta[marker]==0): Ebeta[marker]=0
            elif (delta[marker]==1):
                Ebeta[marker] = sess.run(rnorm(rj/Cj,Sigma2_e/Cj),\
                    feed_dict={colx: x[:,marker].reshape(N,1)})
            # update residuals
            sess.run(epsilon.assign_sub(colx * Ebeta[marker]),\
                feed_dict={colx: x[:,marker].reshape(N,1)})
        # Fullpass over, sample other parameters
        sess.run(NZ.assign(np.sum(delta)))
        sess.run(Pi.assign(sample_pi(M,NZ)))
        sess.run(Sigma2_b.assign(sample_sigma2_b(Ebeta_,NZ,v0B,s0B)),
                feed_dict= {Ebeta_: Ebeta})
        sess.run(Sigma2_e.assign(sample_sigma2_e(N,epsilon,v0E,s0E)))


