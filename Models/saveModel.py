from Models.seqToseqLSTM import *

def saveModel():
    initialModel = regressionModel(learningRate, numNeuron, numStepIn, numStepOut, numFeatures )
    initialModel.save_weights(fileName)
   
    
if __name__ == "__main__":
    learningRate = 0.001
    numNeuron = 30
    numStepIn = 12
    numStepOut = 3
    numFeatures = 2
    fileName = 'In'+str(numStepIn)+'Out'+str(numStepOut)+'Neuron'+str(numNeuron)+'.h5'
    saveModel( learningRate, numNeuron, numStepIn, numStepOut, numFeatures, fileName)
    print('it is done')