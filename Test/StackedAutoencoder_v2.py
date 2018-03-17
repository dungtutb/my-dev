import numpy
import tensorflow as  tf
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data


def autoencode(inputs,hidden_size,layer_name):
    with tf.name_scope(layer_name):
        encoding_layer = tf.layers.dense(inputs,hidden_size,activation=tf.nn.relu,name='encoding_layer_{}'.format(hidden_size))
        output_layer = tf.layers.dense(encoding_layer,inputs.shape[1],name='outputs')
        layer_loss = tf.losses.mean_squared_error(inputs,output_layer)

        all_vars = tf.trainable_variables()

        layer_vars = [v for v in all_vars if v.name.statswish(layer_name)]

        optimizer = tf.train.AdamOptimizer().minimize(layer_loss,var_list=layer_vars,name='{}_opt'.format(layer_name))

        loss_summ = tf.summary.scalar('{}_loss'.format(layer_name),layer_loss)
    return {'input':inputs,'encoding_layer':encoding_layer,'output_layer':output_layer,'layer_loss':layer_loss,'optimizer':optimizer}
    
def train_layer(data,output_layer,layer_loss,optimizer):
    layer_name = output_layer.name.split('/')[0]
    print('Pretraining {}'.format(layer_name))
    num_steps = 1000
    step = 1
    while(step <= num_steps):
        batch = mnist.train.next_batch(batch_size)
        _out_layer, _layer_loss, _ = sess.run([output_layer,layer_loss,optimizer],feed_dict={x:batch[0],y_labels:batch[1]})
        step += 1
    print('layer finished')

batch_size = 100
num_layers = 6
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784], name='x_placeholder')
y_labels = tf.placeholder(tf.float32, shape=[None, 10], name='y_labels_placeholder')

model_layers = []

hidden_size = x.shape[1].value/2
next_layer_inputs = x

for i_layer in range(num_layers):
    layer_name = 'layer_{}'.format(i_layer)
    model_layers.append(autoencode(next_layer_inputs,hidden_size,layer_name))
    next_layer_inputs = model_layers[-1]['encoding_layer']
    hidden_size = int(hidden_size/2)

last_layer = model_layers[-1]
outputs = last_layer['encoding_layer']
with tf.variable_scope('predicted_class'):
    y = tf.layers.dense(outputs,10,activation=tf.nn.softmax)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_labels,logits=y))

global_step = tf.train.get_or_create_global_step()
net_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cross_entropy,global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_labels,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

accuracy_summ = tf.summary.scalar('train_accuracy',accuracy)

#create hooks to pass to the session.  These can be used for adding additional calculations, loggin, etc.
#This hook simply tells the session how many steps to run
hooks=[tf.train.StopAtStepHook(last_step=10000)]

#This command collects all summary ops that have been added to the graph and prepares them to run in the next session
tf.summary.merge_all()

logs_dir = 'logs'



with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=logs_dir,save_summaries_steps=100) as sess:

    # start_time = time.time()

    """First train each layer one at a time, freezing weights from previous layers.
    This was accomplished by declaring which variables to update when each layer optimizer was defined."""
    for layer_dict in model_layers:
        output_layer = layer_dict['output_layer']
        layer_loss = layer_dict['layer_loss']
        optimizer = layer_dict['optimizer']
        train_layer(output_layer, layer_loss, optimizer)

    #Now train the whole network for classification allowing all weights to change.
    while not sess.should_stop():
        batch = mnist.train.next_batch(batch_size)
        _y, _cross_entropy, _net_op, _accuracy = sess.run([y, cross_entropy, net_op, accuracy], feed_dict={x:batch[0],y_labels:batch[1]})
        print(_accuracy)
print('Training complete\n')

#examine the final test set accuracy by loading the trained model, along with the last saved checkpoint
# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('logs\\model.ckpt-10000.meta')
#     new_saver.restore(sess, tf.train.latest_checkpoint('.\\logs'))
#     _accuracy_test = accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y_labels:mnist.test.labels})
#     print('test_set accuracy: {}'.format(_accuracy_test))

# duration = (time.time() - start_time)/60
# print("Run complete.  Total time was {} min".format(duration))