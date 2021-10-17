from Models.seqToseqLSTM import *
from creatingDataset import *
from creatingDataset.defineDataset import *

def trainFunc(dataset, nodeIndex, mergedModelShared,trainedModelShared, learning_rate, num_neuron, n_step_in,n_step_out,n_features, minTime, maxTime, granularity, lag, latNumScaled, lngNumScaled,timeClm, minSize):
    # create x_train and y_train
    #print('defining dataset')
    X_train, y_train =  defineTrainset(dataset,n_step_in,n_step_out,n_features, minTime, maxTime, granularity, lag, latNumScaled, lngNumScaled,timeClm, minSize)
    #print('defining a model and set weight')
    tmpModel = regressionModel(learning_rate, num_neuron, n_step_in, n_step_out,  n_features )
    #tmpModel = regressionModelRMSE(learning_rate, num_neuron, n_step_in, n_step_out,  n_features )
    tmpModel.set_weights(mergedModelShared[nodeIndex])
    #print("Traing:")
    tmpModel.fit(X_train, y_train, batch_size=32, epochs=1, verbose=0)
    #print("Assign the model")
    trainedModelShared[nodeIndex] = tmpModel.get_weights()

