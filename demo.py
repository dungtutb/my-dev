import pandas
import numpy as np
from keras.models import Model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from dbn.models import UnsupervisedDBN
from ..sae.models import StackedAutoEncoder
import h5py
from keras.models import model_from_json

h5f = h5py.File("data.h5","r")
normalized_X = h5f["normalized_X"].value
labeled_Y = h5f["labeled_Y"].value
Y = h5f["Y"].value
Y_A = h5f['Y_A'].value
h5f.close()

n_input = normalized_X.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(normalized_X,Y_A,train_size=0.5,random_state=np.random.seed(7))

#json_file = open("encoder_advanced.json","r")
#encoder_model_json = json_file.read()
#json_file.close()

#encoder_model = model_from_json(encoder_model_json)
#encoder_model.load_weights("model_advanced.h5")
#print("loaded model from disk")

#encoder_model.compile(loss="binary_crossentropy",optimizer="adam")

#X_encoded_train = encoder_model.predict(X_train)
#X_encoded_test = encoder_model.predict(X_test)
dbn = UnsupervisedDBN(hidden_layers_structure=[50,100,200],batch_size=200,learning_rate_rbm=0.06,n_epochs_rbm=10,activation_function="sigmoid")
sae = StackedAutoEncoder(network_architecture=[100,50],batch_size=200,learning_rate_ae=0.001,n_epochs_ae=1,activation_function="sigmoid")

RFClassifier = RandomForestClassifier(n_estimators=25)
classifier_dbn = Pipeline(steps=[('dbn',dbn),('rfc',RFClassifier)])
classifier_dbn.fit(X_train,Y_train)
print("Random Forest Classification using DBN features:\n%s\n" % (metrics.classification_report(Y_test,classifier_dbn.predict(X_test))))

print()
RFClassifier.fit(X_train,Y_train)
print("Random Forest Classification:\n%s\n" % (metrics.classification_report(Y_test,RFClassifier.predict(X_test))))

classifier_sae = Pipeline(steps=[('sae',sae),('rfc',RFClassifier)])
classifier_sae.fit(X_train,Y_train)
print("Random Forest Classification using SAE features:\n%s\n" % (metrics.classification_report(Y_test,classifier_dbn.predict(X_test))))

#RFClassifier.fit(X_encoded_train,Y_train)
#print("Random Forest Classification using Autoecoder features:\n%s\n" % (metrics.classification_report(Y_test,RFClassifier.predict(X_encoded_test))))