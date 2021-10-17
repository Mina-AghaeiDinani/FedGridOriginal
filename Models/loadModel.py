from keras.models import load_model
from Models.seqToseqLSTM import *

def loadModel(mainModel,numNode, learning_rate, num_neuron, n_step_in, n_step_out,  n_features, fileName):
    initialModel = regressionModel(learning_rate, num_neuron, n_step_in, n_step_out,  n_features )   
    initialModel = load_model(fileName)
    for i in range(numNode):
        mainModel[i] = initialModel.get_weights()

def loadModelRMSE(mainModel,numNode, learning_rate, num_neuron, n_step_in, n_step_out,  n_features, fileName):
    initialModel = regressionModelRMSE(learning_rate, num_neuron, n_step_in, n_step_out,  n_features )   
    initialModel = load_model(fileName)
    for i in range(numNode):
        mainModel[i] = initialModel.get_weights()