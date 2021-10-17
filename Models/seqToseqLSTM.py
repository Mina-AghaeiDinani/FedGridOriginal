from keras.models import Sequential
from keras.layers import LSTM , Dropout, Dense, RepeatVector, TimeDistributed # , CuDNNLSTM : sometimes is not working because we need to upgrade keras
#note : from tensorflow 2 we do not nee to specify CuDNNLSTM. we have to nstall CuDNN first
from keras.optimizers import Adam  
#from keras.utils.multi_gpu_utils import  multi_gpu_model
from keras import backend as K
  
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def regressionModelRMSE(learning_rate, num_neuron, n_step_in, n_step_out,  n_features ):
        model = Sequential()
        opt = Adam(learning_rate = learning_rate)
        model.add(LSTM(num_neuron, activation='relu', input_shape=(n_step_in,n_features )))
        model.add(RepeatVector(n_step_out))
        model.add(LSTM(num_neuron, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer = opt, loss = root_mean_squared_error) # 'mse'
        return model
    
    
def regressionModel(learning_rate, num_neuron, n_step_in, n_step_out,  n_features ):
        model = Sequential()
        opt = Adam(learning_rate = learning_rate)
        model.add(LSTM(num_neuron, activation='relu', input_shape=(n_step_in,n_features )))
        #model.add(Dropout(0.2))
        model.add(RepeatVector(n_step_out))
        model.add(LSTM(num_neuron, activation='relu', return_sequences=True))
        #model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer = opt, loss = 'mse') # 'mse'
        return model
    
def regressionModelCuda(learning_rate, num_neuron, n_step_in, n_step_out,  n_features ):
        model = Sequential()
        opt = Adam(learning_rate = learning_rate)
        #model.add(CuDNNLSTM(num_neuron, activation='relu', input_shape=(n_step_in,n_features )))
        model.add(RepeatVector(n_step_out))
        #model.add(CuDNNLSTM(num_neuron, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer = opt, loss = 'mse') # 'mse'
        return model
    

def regressionModelLSTMGPU(learning_rate, num_neuron, n_step_in, n_step_out,  n_features ):
        model = Sequential()
        #opt =tf.optimizers.Adam(learning_rate = learning_rate)
        opt = Adam(learning_rate = learning_rate)
        model.add(LSTM(num_neuron, activation='relu', input_shape=(n_step_in,n_features )))
        #model.add(Dropout(0.2))
        model.add(RepeatVector(n_step_out))
        model.add(LSTM(num_neuron, activation='relu', return_sequences=True))
        #model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(n_features)))
        
        numGPU = 8 # number of GPU that we have
        #model = multi_gpu_model(model , gpus = numGPU)
        model.compile(optimizer = opt, loss = 'mse') # , metrtics =['accuracy']
        
        return model