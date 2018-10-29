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





