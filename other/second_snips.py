###############################
# sample beta
###############################

# New variable definitions
Ebeta = tf.Variable(tf.zeros([M,1]), dtype=tf.float32)
delta = tf.Variable(tf.zeros(M), dtype=tf.float32)
# Placeholders:
Xj = tf.placeholder(tf.float32, shape=(N,1))
ind = tf.placeholder(tf.int32, shape=())

def process_Xj(x_j, error, s2e, s2b, pi, beta_old):
    error = error + (x_j*beta_old)
    Cj = tf_squared_norm(x_j) + s2e/s2b
    rj = tf.tensordot(tf.transpose(x_j), error, 1)[0,0]
    ratio = tf.exp(-( tf.square(rj)/( 2*Cj*s2e )))*tf.sqrt((s2b*Cj)/s2e)
    pij = pi / (pi + ratio*(1-pi))
    toss = rbernoulli(pij)
    def case_zero(): return 0., 0.
    def case_one(): return rnorm(rj/Cj, s2e/Cj), 1.
    beta_new, delta_new = tf.cond(tf.equal(toss,1),case_one, case_zero)
    error = error - (x_j*beta_new)
    return beta_new, delta_new, error

# Computations: 'ta' = to assign 
ta_beta, ta_delta, ta_e = (process_Xj(Xj, e,
                           Sigma2_e, Sigma2_b,
                           Pi, Ebeta[ind,0])) 
ta_nz = tf.reduce_sum(delta)
ta_pi = sample_pi(M,NZ)
ta_s2b = sample_sigma2_b(Ebeta,NZ,v0B,s0B)
ta_s2e = sample_sigma2_e(N,e,v0E,s0E)

# Assignment operations:
# From process_Xj()
beta_item_assign_op = Ebeta[ind,0].assign(ta_beta) 
delta_item_assign_op = delta[ind].assign(ta_delta)
e_up = e.assign(ta_e)
up_grp = (tf.group(beta_item_assign_op,
         delta_item_assign_op, eps_up))
# Rest of parameters
nz_up = NZ.assign(ta_nz)
pi_up = Pi.assign(ta_pi)
s2b_up = Sigma2_b.assign(ta_s2b)
s2e_up = Sigma2_e.assign(ta_s2e)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_iter):
        index = np.random.permutation(M)

        for marker in index:
            feed = {ind: marker, Xj: x[:,[marker]}
            sess.run(up_grp, feed_dict=feed)
        
        sess.run(nz_up)
        sess.run(pi_up)
        sess.run(s2b_up)
        sess.run(s2e_up)