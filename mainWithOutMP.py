import pandas as pd
import os

import time 
from numpy import array
from Models import *
from Models.loadModel import *
from Models.seqToseqLSTM import *
from training import  *
from reproducableResult import *
import time
if __name__ == "__main__":
    
    startTime = time.time()
    reproducableFunc()            
    #%% reading dataset
    df=pd.read_csv(r'Datasets/lux_16to19_CCClean_Add.txt', sep=",")
    #df = df.iloc[:,1:] # check if it is needed
    
    #%% parameters
    time_stamp = 15 #int(sys.argv[0])
    transmission_radius=100 # #int(sys.argv[7])
    valid_max = 300
    validation_time= 300  #  5 minutes = 5 * 60 = 300 seconds  
    
    num_neuron=30 #int(sys.argv[1])
    num_batch=32 #int(sys.argv[2])
    learning_rate=0.001 #float(sys.argv[3])
    num_epoch=1 #int(sys.argv[4])
    n_step_in=12 #int(sys.argv[5])
    n_step_out=3 #int(sys.argv[6])
    n_features=2 #x and y
    

   
    max_lat = 6974.999952 #6950.0   ### notice: we have to write based on what we  used for normalization
    max_lng = 5180.079698 #7299.96
    min_lat = 5975.002277 #5950.0
    min_lng = 4181.04041 #6299.96
    
    Threshold = 0 # int(sys.argv[2])
    ########################### after reading dataset e have to specify these fields
    minTime = 57600 #50400  ######### notice: based on time interval that you want to do simulation you shuold select minimum and maximum of the simulation time
    maxTime = 68400 # 61200
    # "granularity" means how many samples we have in each 5 seconds, we can get it based on the granularity of our system
    # e.g. if we have samples every 100 milliseconds , so number of samples in 5 secs is 50

    # "lag" defines the gap between samples.e.g if granulariy id 100 milliseconds, and the distance betwee samples is 5 seconds. then we want to have 
    # records every 300 miliseconds, so precision here is 3  (50,3) (17,1)
    # we can change lag to decrease, or increse size of dataset

    # "latNum" is the column number of normalized latitude in our dataset,
    # "lngNum" is the column number of normalized longitude in our dataset 
    # "minSize" : how big do you want to select the size of your dataset, to be able to tarin it. 1200 , or 400( it depends on the granularity) 
    # "timeClm" : the title that has been defined for the column time, it must be in string
    granularity = 17
    lag = 1
    latNum = 6
    lngNum = 7
    timeClm = 'time'
    carIdClm = 'carId'
    minSize = 400
    
   
    #%%
    nodesList = df.carId.unique() ##******************************************************************8
    nodesList = nodesList.tolist()
    #%% we have to consider a window for runnig the program
    exeTime = time.time()
 
    current_time = minTime + validation_time 
     #%% at the very beginning they dont have any model to exchange 
        
    nodesNumber =  len(nodesList)
  
  
    # we consider two diffrent models one for training and one for main models 
    #because they might be async, it means one node can merge its received models 
    # while another neighbor want to use ex model 
    #we have to save main model of each node
    trainedModel = [ [
             array([[0 for _ in range(num_neuron*4)] for _ in range(n_features)]),
             array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
             array([0 for _ in range(num_neuron*4)]),
             array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
             array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
             array([0 for _ in range(num_neuron*4)]),
             array([[0 for _ in range(n_features)] for _ in range(num_neuron)]), # 50 is the number of classes
             array([0 for _ in range(n_features)])
             ]   for _ in range(nodesNumber)]
  
    # to avoid conflict we define another model for merging
    mergedModel = [ [
             array([[0 for _ in range(num_neuron*4)] for _ in range(n_features)]),
             array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
             array([0 for _ in range(num_neuron*4)]),
             array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
             array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
             array([0 for _ in range(num_neuron*4)]),
             array([[0 for _ in range(n_features)] for _ in range(num_neuron)]), # 50 is the number of classes
             array([0 for _ in range(n_features)])
             ]   for _ in range(nodesNumber)]
       
     # for each model we consider the number of involved samples in trainig
    numSamplesTrainedModel = [0 for _ in range(nodesNumber)]
    numSamplesMergedModel  = [0 for _ in range(nodesNumber)]   
   
    
    ########################################################### Initialization
    ##to have reproducable results we need to have the same initialization all the time
    fileNameModel = '/home/mina/Project/12to3FitMSe.h5'
    loadModel(mergedModel,nodesNumber, learning_rate, num_neuron, n_step_in, n_step_out,  n_features, fileNameModel )
    print('Initialization with the same weight has been done.')
    nodesList = nodesList[:600]
    ####################################### Train on local dataset for each node
    for nodeId in nodesList:

        dfTmp = df[df[carIdClm] ==  nodeId]
        dfTmp = dfTmp[ (dfTmp[timeClm] <= current_time) | (dfTmp[timeClm] > maxTime)]
        nodeIndex = nodesList.index(nodeId)    
        trainFunc( dfTmp, nodeIndex, mergedModel, trainedModel, learning_rate , num_neuron, n_step_in,n_step_out, n_features, minTime, maxTime, granularity, lag, latNum, lngNum,timeClm, minSize)

    print('Code is ended at:' , time.time() - startTime)
    '''
    ###################################### We have to do Training and merging repeatedly in a loop
    turn = 0
    ##################################################################
    #################### these steps must be reapeted#################
    while (current_time < maxTime - 60):            
        current_time = current_time + time_stamp
        turn = turn +1
        print('turn:', turn, 'current time:', current_time)
        print("spent time: %f" %(time.time() - exeTime))
        ### only nodes who are in the current time can exchange the models
        
        df_existed_nodes = df[df[timeClm] == current_time]
        current_nodes_list = df_existed_nodes.carId.unique() #**************************************************************
        
        ####################################################################################
        ##################################Merging###########################################
        ####################################################################################        
        processLst = [] 
        
        for nodeId in current_nodes_list:     
             
             if (len(processLst) == n_process):
                     [ p.join() for p in processLst]
             processLst = []
             
             df_current = df[df[timeClm] == current_time]
             df_main = df[df[carIdClm] == nodeId] # we need local dataset to create  validation set
             nodeIndex = nodesList.index(nodeId)
             model.set_weights(mainModel_train[nodeIndex]) # we will also send the model to do evaluation            
                       
                 
             p = multiprocessing.Process(target= mergeFunc , args= (df_main, df_current, mergedModelShared, trainedModelShared,  nodeId, nodeIndex,current_time,lock   ))            
             p.start()
             processLst.append(p) 
                     
        [ p.join() for p in processLst]           
        processLst = []       
        '''