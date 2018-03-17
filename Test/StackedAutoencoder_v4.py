import tensorflow as tf
import numpy as np
from functools import partial
import sys
import h5py
#import matplotlib.pyplot as plt
#(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = keras.datasets.mnist.load_data()

def train_autoencoder(X_train, n_hidden,n_epochs,batch_size,learning_rate = 0.01, l2_reg=0.0005, seed=42,hidden_activation=tf.nn.elu,output_activation=tf.nn.elu):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        
        n_inputs = X_train.shape[1]
        
        X = tf.placeholder(tf.float32,shape=[None,(n_inputs)])

        my_dense_layer = partial(tf.layers.dense,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        hidden = my_dense_layer(X,n_hidden,activation=hidden_activation,name = "hidden")
        outputs = my_dense_layer(hidden,n_inputs,activation=output_activation,name="outputs")

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_opt = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train) // batch_size
            for iter in range(n_batches-1):
                #print("\r{}%".format(100 * iter // n_batches), end="")
                sys.stdout.flush()    
                X_batch = X_train[batch_size * iter : batch_size * (iter + 1)]
                sess.run(training_opt,feed_dict={X:X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X:X_train})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
        params = dict([(var.name,var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

        hidden_val = hidden.eval(feed_dict={X:X_train})
        return hidden_val,params["hidden/kernel:0"],params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"]

h5f = h5py.File('data.h5','r')

normalized_X = h5f['normalized_X'].value

h5f.close()

hidden_output, W1, b1, W4, b4 = train_autoencoder(normalized_X,n_hidden=100,n_epochs=1,batch_size=150,output_activation=None)

print("asdasdsad")