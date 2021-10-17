from creatingDataset.defineDataset import *
from Models.seqToseqLSTM import  *
import math
import os
import sys
from numpy import array

def calculateDist(x1,y1,x2,y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def mergeFunc( df_main,df_current, mergedModelShared, trainedModelShared,  nodeId, nodeIndex,current_time,lock , learning_rate,
              n_step_in, n_step_out, n_features, num_neuron, minTime, maxTime, granularity, lag, latNum, lngNum, carIdNum 
             , latNumScaled, lngNumScaled, timeClm, carIdClm, minSize, fileNameSample, 
              Threshold, transmission_radius,nodesList):  
         
         summationModel = [
         array([[0 for _ in range(num_neuron*4)] for _ in range(n_features)]),
         array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
         array([0 for _ in range(num_neuron*4)]),
         array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
         array([[0 for _ in range(num_neuron*4)] for _ in range(num_neuron)]),
         array([0 for _ in range(num_neuron*4)]),
         array([[0 for _ in range(n_features)] for _ in range(num_neuron)]), 
         array([0 for _ in range(n_features)])
         ] 
         #print('Merging function')   
         
         #split dataset to validation set        
         X_valid, y_valid = defineValidationset(df_main,n_step_in,n_step_out,n_features, minTime, maxTime, granularity, lag, latNumScaled, lngNumScaled,timeClm, minSize)
         #print(X_valid.shape, y_valid.shape)
         
         # Current position of main node
         df_main_node = df_current [df_current[carIdClm] == nodeId ]
         lat_main = df_main_node['lat'].values[0] #df_main_node.iloc[0,2]  ###chexk beshe
         lng_main = df_main_node['lng'].values[0] #df_main_node.iloc[0,3]  ###
         #print(lat_main, '  ', lng_main)
         
         loss_lst =[]
         visited_lst = []     
         invLossLst = list()
         
         
         #define a model
         tmpModel = regressionModel(learning_rate, num_neuron, n_step_in, n_step_out,  n_features )
         tmpModel.set_weights(trainedModelShared[nodeIndex])
         #print('weights were set')
         
         mainLoss = tmpModel.evaluate(X_valid, y_valid, verbose=0)
         #print('evaluation done:' , mainLoss)
         
         if (mainLoss == 0):
            invLoss = 6
         elif (mainLoss == 1):
            invLoss = 0.004
         elif (mainLoss < 1):
            invLoss = -1 * (math.log10(mainLoss))
         elif (mainLoss > 1):
            invLoss = (math.log10(mainLoss))
       
         ##print(mainLoss)
         loss_lst.append(mainLoss)      
         invLossLst.append(invLoss)
         visited_lst.append(nodeId)
        
         #Then we have to see how many neighbors this node has
         df_neighbor_nodes = df_current[ df_current[carIdClm] != nodeId ]
         #print('number of neighbors: ' , len(df_neighbor_nodes))
                     
         for i in range(len(df_neighbor_nodes)):
            # position of other existing nodes
            lat_neighbor = df_neighbor_nodes.iloc[i,latNum] 
            lng_neighbor = df_neighbor_nodes.iloc[i,lngNum]
            Id_neighbor = df_neighbor_nodes.iloc[i,carIdNum]
            
            neighborIndex = nodesList.index(Id_neighbor) 
            #print('neighbor Index:' , neighborIndex)
            
              
            # Here we have coordinates, If it was lat and lng I have to use 
            # Compute the distance of neighbor with the exsiting node
            distance = calculateDist(lat_main,lng_main,lat_neighbor,lng_neighbor)
            
            if (distance <= transmission_radius):
                
                # we have to evaluate its model to find the amount of eror
                tmpModel.set_weights(trainedModelShared[neighborIndex])
                neighborLoss = tmpModel.evaluate(X_valid, y_valid, verbose=0) 
                #print('The node is in contact, with loss:', neighborLoss )
               
                if (neighborLoss== 0):
                    invLoss = 6
                elif (neighborLoss == 1):
                    invLoss = 0.004
                elif (neighborLoss < 1):
                    invLoss = -1 * (math.log10(neighborLoss))
                elif (neighborLoss > 1):
                    invLoss = (math.log10(neighborLoss))
       
                ##print(mainLoss)
                loss_lst.append(neighborLoss)
                invLossLst.append(invLoss)
                visited_lst.append(Id_neighbor)
                
         totalLoss = sum(loss_lst)
         totalLossInv = sum(invLossLst)
         #print( 'total loss and its inverse: ' , totalLoss, '  ', totalLossInv)
        
        
        
         ##*******************************************
         #compute the n percent
         nPercent = Threshold * totalLossInv / 100
         # sort the list
         sortedSamples = sorted(invLossLst)
         ##*******************************************
         #################################   main part #################################
         k = len(sortedSamples) - 2
         flg = True 
         deletedLst = list()
         while ( ( k >= 0) and (flg == True)):
            summation = 0
            deletedLst = list()
            for i in range(k+1):
                summation += sortedSamples[i]
            if (summation < nPercent):
                for i in range(k+1):
                    deletedLst.append(sortedSamples[i])
                    flg = False
            k = k - 1 
         ##############################################################################
        
         numExModel = len(invLossLst)
         sizeExModel = totalLossInv
        
         #print('Size of ex models: ' ,numExModel )
        
         # we have to also consider the main model to build a new meta model
         for elem in deletedLst:
            idx = invLossLst.index(elem)
            visited_lst.pop(idx)
            invLossLst.pop(idx)
        
         totalLossInv = sum(invLossLst)
         #print('Size of Involved models: ' , len(invLossLst) )
         ##############################################################
         # After exchanging models we can merge them
         for Indx, neighborId in enumerate(visited_lst):
            neighborIndex = nodesList.index(neighborId)
            multiplication = invLossLst[Indx] / totalLossInv
            
            
            for layer in range(8):
                summationModel[layer] = summationModel[layer] + multiplication * trainedModelShared[neighborIndex][layer]

         # we have to update the size of new model and also new model
         #for layer in range(8):
         #   mergedModelShared[nodeIndex][layer] = summationModel[layer]  
         tmpModel = regressionModel(learning_rate, num_neuron, n_step_in, n_step_out,  n_features )
         tmpModel.set_weights(summationModel)
         mergedModelShared[nodeIndex] = tmpModel.get_weights()
        
         lock.acquire()
         try:
                log_samples = open(fileNameSample , 'a+')
                log_samples.write('%f \t %s \t %f  \t %d \t %f  \t %d \n' %(current_time, nodeId, totalLossInv, len(invLossLst) , sizeExModel , numExModel))     
                log_samples.close()
         finally:
                lock.release()   
       

         #end of critical section
         