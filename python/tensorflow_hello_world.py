#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:29:20 2018

@author: admin
"""

import edward 

import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")

sess = tf.Session()

print(sess.run(hello))
#we get b'Hello, TensorFlow!'
#we should get Hello, TensorFlow!

a = tf.constant(10)
 
b = tf.constant(100)
 
print(sess.run(a + b))
#110
 
In [8]: c = a * b
 
In [11]: with tf.Session() as sess:
   ....:     print sess.run(c)
   ....:     print c.eval()
   ....:     
1000
1000
In [16]: tf.InteractiveSession()
Out[16]: <tensorflow.python.client.session.InteractiveSession at 0x7fe6547d3750>
 
In [17]: a = tf.zeros((2,2)); b = tf.ones((2,2))
 
In [18]: tf.reduce_sum(b, reduction_indices=1).eval()
Out[18]: array([ 2.,  2.], dtype=float32)
 
In [19]: a.get_shape()
Out[19]: TensorShape([Dimension(2), Dimension(2)])
 
In [20]: tf.reshape(a, (1,4)).eval()
Out[20]: array([[ 0.,  0.,  0.,  0.]], dtype=float32)