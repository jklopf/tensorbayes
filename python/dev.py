#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:30:12 2018

@author: admin
"""

import numpy as np
import tensorflow as tf


# Get the numbers of columns in the csv:

csv_in = open("ex.csv", "r")                        # open the csv
ncol = len(csv_in.readline().split(","))            # read the first line and count the # of columns
csv_in.close()                                      # close the csv
print("Number of columns in the csv: " + str(ncol)) # print the # of columns


# Create random column order list (dataset) + iterator
col_list = tf.data.Dataset.range(ncol).shuffle(buffer_size=ncol)
col_next = col_list.make_one_shot_iterator().get_next()

def scale_zscore(vector):
    mean, var = tf.nn.moments(vector, axes=[0])
    normalized_col = tf.map_fn(lambda x: (x - mean)/tf.sqrt(var), vector)
    return normalized_col

# Launch of graph
# =============================================================================
# with tf.Session() as sess:
# 
#     while True: # Loop on 'col_next', the queue of column iterator
#         try:
#             index = sess.run(col_next)
#             dataset = tf.contrib.data.CsvDataset( # Creates a dataset of the current csv column
#                         "ex.csv",
#                         [tf.float32],
#                         select_cols=[index]  # Only parse last three columns
#                     )
#             next_element = dataset.make_one_shot_iterator().get_next() # Creates an iterator
#             print('Current column to be full pass: ' + str(index))
#             current_col = []
#             while True: 
#                 try:
#                     current_col.append(sess.run(next_element)[0]) # Full pass
#                 except tf.errors.OutOfRangeError: # End of full pass
#                     print(current_col)
#                     normalized_col =  sess.run(scale_zscore(tf.stack(current_col)))
#                     print(normalized_col)
#                     
#                     break
# 
# 
#             
# 
#         except tf.errors.OutOfRangeError:
#             break
# 
# 
# =============================================================================

sess = tf.Session()

while True:
    # Loop on 'col_next', the queue of column iterator
    try:
        index = sess.run(col_next)
        dataset = tf.contrib.data.CsvDataset( # Creates a dataset of the current csv column
                    "ex.csv",
                    [tf.float32],
                    select_cols=[index]  # Only parse last three columns
                )
        next_element = dataset.make_one_shot_iterator().get_next() # Creates an iterator
        print('Current column to be full pass: ' + str(index))
        current_col = []
        while True: 
            try:
                current_col.append(sess.run(next_element)[0]) # Full pass
            except tf.errors.OutOfRangeError: # End of full pass
                print(current_col)
                normalized_col =  sess.run(scale_zscore(tf.stack(current_col)))
                print(normalized_col)
                
                break
    
    

toy = tf.contrib.data.CsvDataset(
        # Creates a dataset of the current csv column
        "ex.csv",
        [tf.float32],
        select_cols=[0]  # Only parse last three columns
        )

a = np.random.choice(len(x_vals), size=batch_size)

aa = np.transpose([x_vals[rand_index]])



sess.close()


from __future__ import print_function
#import tensorflow as tf

def tf__ftemp(eps, beta, x):
    global Sigma2_e
    try:
        with tf.name_scope('ftemp'):
            
              index = np.random.permutation(M)
              epsilon = eps
              Ebeta = beta
              ny = 0

              def extra_test(ny_2, epsilon_1, Ebeta_3):
                    with tf.name_scope('extra_test'):
                          return True

              def loop_body(loop_vars, ny_2, epsilon_1, Ebeta_3):
                with tf.name_scope('loop_body'):
                  marker = loop_vars
                  epsilon_1 = epsilon_1 + x[:, (marker)] * ag__.get_item(Ebeta_3,
                      marker, opts=ag__.GetItemOpts(element_dtype=None))
                  toss, Cj, rj = marker_toss(x[:, (marker)])

                  def if_true():
                    with tf.name_scope('if_true'):
                      Ebeta_1, = Ebeta_3,
                      Ebeta_1 = ag__.set_item(Ebeta_1, marker, 0)
                      return ny_2, Ebeta_1

                  def if_false():
                    with tf.name_scope('if_false'):
                      ny_1, Ebeta_2 = ny_2, Ebeta_3
                      Ebeta_2 = ag__.set_item(Ebeta_2, marker, rnorm(rj/Cj,Sigma2_e/Cj))
                      ny_1 += 1
                      return ny_1, Ebeta_2
                  ny_2, Ebeta_3 = ag__.utils.run_cond(tf.equal(toss, 0), if_true,
                      if_false)
                  epsilon_1 = epsilon_1 - x[:, (marker)] * ag__.get_item(Ebeta_3,
                      marker, opts=ag__.GetItemOpts(element_dtype=None))
                  return ny_2, epsilon_1, Ebeta_3
              ny, epsilon, Ebeta = ag__.for_stmt(index, extra_test, loop_body, (ny,
                  epsilon, Ebeta))
              return epsilon, Ebeta, ny
    except:
        ag__.rewrite_graph_construction_error(ag_source_map__)


