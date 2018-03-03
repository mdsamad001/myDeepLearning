#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:51:49 2017

@author: mdsamad
"""

# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
#from __future__ import division, print_function, absolute_import


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import random

#from sklearn.preprocessing import StandardScaler, MinMaxScalar

from sklearn import preprocessing


#from sklearn.preprocessing import OneHotEncoder

mnist = fetch_mldata('MNIST original')
mdata = mnist.data
label = mnist.target
fdata = np.c_[mdata,label]

# Shuffling over lists
random.shuffle(fdata)

Xdata = np.array (fdata)

Xdat = Xdata[:,:-1]
y = Xdata[:,-1]

x_train,x_rest,y_train,y_rest = train_test_split(Xdat,y,train_size=0.7) 

# Test is now the validation so used train_size =0.66, otherwise it would be 0.33
x_test,x_val, y_test, y_val = train_test_split(x_rest,y_rest, train_size=0.66) 


# Min-max Normalize 
min_max_scaler = preprocessing.MinMaxScaler()
trainX = min_max_scaler.fit_transform(x_train)

testX = min_max_scaler.fit_transform(x_test)



# Z-score
#scaler = preprocessing.StandardScaler().fit(x_train)
#trainX = scaler.transform(x_train)


batch_size = 700
batch_num = np.floor(len(trainX)/batch_size)

batch_data = np.split(trainX,batch_num)
batch_label =np.split(y_train,batch_num)

batch_data = np.array(batch_data)
batch_label = np.array(batch_label)



print(batch_data.shape)

#

# Parameters
learning_rate = 0.05
training_epochs = 800
#batch_size = 265
display_step = 1
#examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_hidden_3 = 64
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),

}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes])),
    
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    
    # Output, class prediction
    out = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    
    
    return out


# Building the decoder
#def decoder(x):
#    # Encoder Hidden layer with sigmoid activation #1
#    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
#                                   biases['decoder_b1']))
#    # Decoder Hidden layer with sigmoid activation #2
#    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
#                                   biases['decoder_b2']))
#    
#    # Decoder Hidden layer with sigmoid activation #2
#    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
#                                   biases['decoder_b3']))
#    
#    
#    return layer_3

    
    
# Construct model
#pred = conv_net(x, weights, biases, keep_prob)
#encoder_op = encoder(X)

pred = encoder(X)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

############################    
# Construct model
#encoder_op = encoder(X)
#decoder_op = decoder(encoder_op)

# Prediction
#y_pred = decoder_op
# Targets (Labels) are the input data.
#y_true = X

# Define loss and optimizer, minimize the squared error
#cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(batch_num)#int(mnist.train.num_examples/batch_size)
    # Training cycle
   # b_iter = 0
    trAcc = []
    tsAcc = []
    for epoch in range(training_epochs):
        # Loop over all batches
        for b_iter in range(total_batch):
           # print(b_iter)
            
            batch_x = batch_data[b_iter,:,:]
            a = batch_label [b_iter,:]
            a = a.astype('int')
            batch_y = np.zeros((len(a),n_classes))
            batch_y [np.arange(len(a)),a]=1
                     
                     # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
        
        if epoch % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                              y: batch_y})
            trAcc.append(acc)
            print("Epoch: " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                 
            
            b = y_test 
            b = b.astype('int')
            testy = np.zeros((len(b),n_classes))
            testy[np.arange(len(b)),b]=1
  
            #print("Testing Accuracy:", \
            #Tacc  =   sess.run(accuracy, feed_dict={X: testX, y: testy}))
            
            _,Tacc  =   sess.run([cost,accuracy], feed_dict={X: testX, y: testy})
            tsAcc.append(Tacc)
            
           # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
           # _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
           # b_iter +=1
            
        #After each epoch 
        
        
        # Display logs per epoch step
       # if epoch % display_step == 0:
        #    print("Epoch:", '%04d' % (epoch+1),
         #         "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    fig = plt.figure()
    plt.plot(trAcc,'o-')
    plt.plot(tsAcc,'*-')

    # Applying encode and decode over test set
    #encode_decode = sess.run(
     #   y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
   
    b = y_test 
    b = b.astype('int')
    testy = np.zeros((len(b),n_classes))
    testy[np.arange(len(b)),b]=1

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: testX,
                                      y: testy}))
filename='l_0_005_SGD_256_128_64_relu_9800_SGD.jpg'

fig.savefig(filename,dpi=100) 