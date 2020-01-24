a = tf.ones([2,2])
b = tf.reduce_sum(a, name="sum_of_matrix_elements")
print(a)
> Tensor("ones:0", shape=(2, 2), dtype=float32)
print(b)
> Tensor("sum_of_matrix_elements:0", shape=(), dtype=float32)

###################################################################
# sampling functions
###################################################################

def sample_sigma2_b(betas, NZ, v0B, s0B):
    # sample variance of betas
    df = v0B+NZ
    scale = (tf_squared_norm(betas)+v0B*s0B) / df  
    sample = rinvchisq(df, scale)
    return sample

def sample_sigma2_e(N, epsilon, v0E, s0E):
    # sample variance of error
    df = v0E + N
    scale = (tf_squared_norm(epsilon) + v0E*s0E) / df
    sample = rinvchisq(df, scale)
    return sample

def sample_pi(M, mNZ):
    # sample mixture weight
    sample = rbeta(mNZ+1, M-mNZ+1)
    return sample

###################################################################
# Parameter initialization
###################################################################


# Variables:
# Vector of sampled marker effect sizes
Ebeta = tf.Variable(tf.zeros([M,1], dtype=tf.float32), dtype=tf.float32)
# Vector of marker model inclusion draws
Ny = tf.Variable(tf.zeros(M, dtype=tf.float32), dtype=tf.float32)
# Number of included marker in the model
NZ = tf.Variable(0., dtype=tf.float32)
# Mixture weight (Pi)
Ew = tf.Variable(0.5, dtype=tf.float32)
# Vector of the error
epsilon = tf.Variable(Y, dtype=tf.float32)
# Variance of the error
Sigma2_e = tf.Variable(tf_squared_norm(Y) / (N*0.5), dtype=tf.float32)
# Variance of the marker effect sizes
Sigma2_b = tf.Variable(rbeta(1., 1.), dtype=tf.float32)
# Constants:
# degrees of freedom of scaled-inv-chi2 prior
v0E = tf.constant(4.0, dtype=tf.float32)
v0B = tf.constant(4.0, dtype=tf.float32)
# scale factor of scaled-inv-chi2 prior
s0B = Sigma2_b.initialized_value() / 2
s0E = Sigma2_e.initialized_value() / 2