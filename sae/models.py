import numpy
import tensorflow as tf
from functools import partial
from .activations import SigmoidActivationFunction, ReLUActivationFunction
import sys
import numpy as np
from .utils import batch_generator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

class BaseModel(object):
    def save(self, save_path):
        import pickle

        with open(save_path, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, load_path):
        import pickle

        with open(load_path, 'rb') as fp:
            return pickle.load(fp)

class AutoEncoder(BaseModel):
    """
    This class implements a autoencoder based on TensorFlow.
    """

    def __init__(self,
            n_hidden_units=100,
            activation_function='sigmoid',
            optimization_algorithm='sgd',
            learning_rate=1e-3,
            l2_reg=0.0005,
            n_epochs=10,
            batch_size=32,
            verbose=True):
        self.n_hidden_units = n_hidden_units
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        if self.activation_function == 'sigmoid':   
            self._activation_function_class = SigmoidActivationFunction
        elif self.activation_function == 'relu':
            self._activation_function_class = ReLUActivationFunction
    
    def fit(self,_data):
        graph = tf.Graph()
        with graph.as_default():
            n_input = _data.shape[1]
        
            X = tf.placeholder(tf.float32,shape=[None,(n_input)])

            my_dense_layer = partial(tf.layers.dense,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
        
            hidden = my_dense_layer(X,self.n_hidden_units,activation=self.activation_function,name="hidden")
            outputs = my_dense_layer(hidden,n_input,activation=self.activation_function,name="outputs")
        
            reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([reconstruction_loss] + reg_losses)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            training_opt = optimizer.minimize(loss)

            init = tf.global_variables_initializer()
        with tf.Session(graph=graph) as sess:
            init.run()
            for iteration in range(1, self.n_epochs + 1):
                idx = np.random.permutation(len(_data))
                data = _data[idx]
                for X_batch in batch_generator(self.batch_size, data):
                    sess.run(training_opt,feed_dict={X:X_batch})
                if self.verbose:
                    error = reconstruction_loss.eval(feed_dict={X:data})
                    print(">> Epoch %d finished \tAE Reconstruction error %f" % (iteration, error))
        params = dict([(var.name,var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        self.weights = params["hidden/kernel:0"]
        self.bias = params["hidden/bias:0"]

        return self
    
    def transform(self, X):
        
        if len(X.shape) == 1:  # It is a single sample
            return self._compute_hidden_units(X)
        transformed_data = self._compute_hidden_units_matrix(X)
        return transformed_data
    
    def _compute_hidden_units(self, vector_visible_units):
        """
        Computes hidden unit outputs.
        :param vector_visible_units: array-like, shape = (n_features, )
        :return:
        """
        v = np.expand_dims(vector_visible_units, 0)
        return np.squeeze(self._compute_hidden_units_matrix(v))

    def _compute_hidden_units_matrix(self, matrix_visible_units):
        """
        Computes hidden unit outputs.
        :param matrix_visible_units: array-like, shape = (n_samples, n_features)
        :return:
        """
        return np.transpose(self._activation_function_class.function(
            np.dot(self.weights, np.transpose(matrix_visible_units)) + self.bias[:, np.newaxis]))

class StackedAutoEncoder(BaseModel):
    def __init__(self,
            network_architecture=[100,100],
            activation_function='sigmoid',
            optimization_algorithm='sgd',
            learning_rate_ae=1e-3,
            l2_reg=0.0005,
            n_epochs_ae=10,
            batch_size=32,
            verbose=True):

        #self.n_input = network_architecture[0]
        self.network_architecture = network_architecture
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        #self.n_layers = len(self.network_architecture)
        self.learning_rate_ae = learning_rate_ae
        self.l2_reg_ae = l2_reg
        self.n_epochs_ae = n_epochs_ae
        self.batch_size = batch_size
        self.sae_layers = None
        self.encoding = None
        self.verbose = verbose
        self.ae_class = AutoEncoder

    def fit(self,X):
        self.n_input = X.shape[1]
        self.sae_layers = list()

        for n_hidden_units in self.network_architecture:
            ae = self.ae_class(n_hidden_units=n_hidden_units,
                activation_function=self.activation_function,
                optimization_algorithm=self.optimization_algorithm,
                learning_rate=self.learning_rate_ae,
                l2_reg=self.l2_reg_ae,
                n_epochs=self.n_epochs_ae,
                batch_size=self.batch_size,
                verbose=self.verbose)
            self.sae_layers.append(ae)
        
        if self.verbose:
            print("[START] Pre-training step:")
        input_data = X
        for ae in self.sae_layers:
            ae.fit(input_data)
            input_data = ae.transfrom(input_data)
        if self.verbose:
            print("[END] Pre-training step")
        
        return self

    def transform(self, X):
        input_data = X
        for ae in self.sae_layers:
            input_data = ae.transform(input_data)
        
        return input_data
    
import h5py

h5f = h5py.File("data.h5","r")
normalized_X = h5f["normalized_X"].value
labeled_Y = h5f["labeled_Y"].value
Y = h5f["Y"].value
Y_A = h5f['Y_A'].value
h5f.close()

X_train, X_test, Y_train, Y_test = train_test_split(normalized_X,Y_A,train_size=0.5,random_state=np.random.seed(7))
sae = StackedAutoEncoder(network_architecture=[100,50],batch_size=200,learning_rate_ae=0.001,n_epochs_ae=1,activation_function="sigmoid")
RFClassifier = RandomForestClassifier(n_estimators=25)

classifier_sae = Pipeline(steps=[('sae',sae),('rfc',RFClassifier)])
classifier_sae.fit(X_train,Y_train)
print("Random Forest Classification using SAE features:\n%s\n" % (metrics.classification_report(Y_test,classifier_sae.predict(X_test))))

