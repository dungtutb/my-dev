import tensorflow as tf
import h5py

h5f = h5py.File('data.h5', 'r')
normalized_X = h5f['normalized_X'].value
#labeled_Y = h5f['labeled_Y'].value
#Y = h5f['Y'].value
#Y_A = h5f['Y_A'].value
h5f.close()

n_inputs = normalized_X.shape[1]

n_hidden1 = 100

n_outputs = n_inputs

def kl_divergence(p,q):
    return p * tf.log(p/q) + (1-p) * tf.log((1-p) * (1-q))

learning_rate = 0.001
sparsity_target = 0.1
sparsity_weight = 0.2

X = tf.placeholder(tf.float32, shape=[None,n_inputs])

hidden1 = tf.layers.dense(X,n_hidden1,activation=tf.nn.sigmoid)
outputs = tf.layers.dense(hidden1,n_outputs)

hidden1_mean = tf.reduce_mean(hidden1, axis=0) # batch mean
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE
loss = reconstruction_loss + sparsity_weight * sparsity_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 100
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batchs = (int)(normalized_X.shape[0] / batch_size)

        for iteration in range(n_batchs):
            X_batch = normalized_X[iteration*batch_size:(iteration+1)*batch_size]
            sess.run(training_op, feed_dict={X: X_batch})
        reconstruction_loss_val, sparsity_loss_val, loss_val = sess.run([reconstruction_loss, sparsity_loss, loss], feed_dict={X: X_batch})

        print("\r{}".format(epoch), "Train MSE:", reconstruction_loss_val, "\tSparsity loss:", sparsity_loss_val, "\tTotal loss:", loss_val)
    saver.save(sess, "./my_model_sparse.ckpt")