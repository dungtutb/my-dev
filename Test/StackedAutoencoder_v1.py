#nguon https://github.com/PaulZhutovsky/3dconv-autoencoder/blob/master/3dconv_autoencoder.py
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import h5py

class Stacked_AE(object):
    def __init__(self, network_architecture=[10,10], activation='sigmoid'):

        self.n_input = network_architecture[0]
        self.network_architecture = network_architecture
        self.activation = activation
        self.n_layers = len(self.network_architecture)
        self.layers = []
        self.encoding = None
        self.autoencoder = None

        self.build_network()
    
    def build_network(self):
        input_layer = Input(shape = (self.n_input,),name='input_layer')

        #encoding
        x = input_layer
        for i_layer in range(1,self.n_layers):
            n_hidden = self.network_architecture[i_layer]
            x = self.build_layer_block_encoding(x,n_hidden,i=i_layer)
        encoding_output = x

        self.encoding = Model(inputs=input_layer,outputs=encoding_output)

        #decoding
        for i_layer in range(1,self.n_layers):
            n_hidden = self.network_architecture[-(i_layer+1)]
            x = self.build_layer_block_decoding(x,n_hidden, i=i_layer)
        

        self.autoencoder = Model(inputs=input_layer,outputs=x)
   
    def build_layer_block_encoding(self,input_layer, n_hidden, i=1):
        encoding = Dense(n_hidden,activation=self.activation,name='enc_layer{}'.format(i))
        self.layers.append(encoding)
        return encoding(input_layer)
    
    def build_layer_block_decoding(self,input_layer, n_hidden, i=1):
        decoding = Dense(n_hidden,activation=self.activation,name='dec_layer{}'.format(i))
        return decoding(input_layer)
    
    def summary(self):
        self.autoencoder.summary()
    
    def compile(self,loss_function='binary_crossentropy',optimization_algorithm='adam'):
        self.autoencoder.compile(loss=loss_function,optimizer=optimization_algorithm)
    
    
h5f = h5py.File('data.h5', 'r')
normalized_X = h5f['normalized_X'].value
#labeled_Y = h5f['labeled_Y'].value
#Y = h5f['Y'].value
#Y_A = h5f['Y_A'].value
h5f.close()

n_input = normalized_X.shape[1]
network_architecture = [n_input,100,30]
sae = Stacked_AE(network_architecture=network_architecture,activation='sigmoid')
sae.compile()
sae.summary()
sae.autoencoder.fit(normalized_X,normalized_X,epochs=1,batch_size=200)
for layer in sae.layers:
    g = layer.get_config()
    h = layer.get_weights()
    print(g)
    print(h)
    layer.use_bias