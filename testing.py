from creatingDataset.defineDataset import *
from Models.seqToseqLSTM import  *
import math
import pandas as pd

def calculateDist(x1,y1,x2,y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def testFunc(df_testT ,nodeId , nodeIndex,mergedModelShared, current_time_testT, end_time_testT, min_timeT, lock,learning_rate, n_step_in, n_step_out, n_features, num_neuron, time_stamp, granularity ,
             timeClm , timeNum,latNumScaled,lngNumScaled, max_lat, min_lat, max_lng, min_lng,fileNameResult):
    
    #print('I am in the test function')
            
    inputSeq = n_step_in * granularity
    outputSeq = n_step_out * granularity
    
    dfStart = df_testT[df_testT[timeClm] <= current_time_testT]
    dfEnd = df_testT[df_testT[timeClm] > current_time_testT]
    
    for i in range(time_stamp): # exchange time 
        dfEnd1 = dfEnd.iloc[:outputSeq+(i*3),:]  #################### we want to have testing every 1 second
        dfStart1 = dfStart.iloc[-inputSeq+(i*3):,:] # dfSrart.tail(inputSeq)
        df_testShort = pd.concat([dfStart1,dfEnd1])
        if (len(df_testShort) >= (inputSeq+outputSeq )):
            timeTest = df_testShort.iloc[inputSeq , timeNum]
            ##print('time in test set', timeTest)
            
            #define X,y test set
            X_test_short, y_test_short = defineShortTestset(df_testShort,n_step_in,n_step_out,n_features,granularity,latNumScaled,lngNumScaled)
            #print(X_test_short.shape , y_test_short.shape)
            
            #we have to rescale the results 
            # then mius them from each other 
            # to get how far they are from each other        
            lat_real_short5  =  y_test_short[0][0][0] * (max_lat - min_lat) + min_lat 
            lng_real_short5  =  y_test_short[0][0][1] * (max_lng - min_lng) + min_lng
            lat_real_short10 =  y_test_short[0][1][0] * (max_lat - min_lat) + min_lat
            lng_real_short10 =  y_test_short[0][1][1] * (max_lng - min_lng) + min_lng
            lat_real_short15 =  y_test_short[0][2][0] * (max_lat - min_lat) + min_lat
            lng_real_short15 =  y_test_short[0][2][1] * (max_lng - min_lng) + min_lng
            #print(" real long:" , lng_real_short15)
            
                
             # predict the next cells   
            tmpModel = regressionModel(learning_rate, num_neuron, n_step_in, n_step_out,  n_features )
            tmpModel.set_weights(mergedModelShared[nodeIndex])
            pre_result = tmpModel.predict(X_test_short)
            #pre_result = tmpModel(X_test_short)
            lat_predicted5  = pre_result[0][0][0] * (max_lat - min_lat) + min_lat
            lng_predicted5  = pre_result[0][0][1] * (max_lng - min_lng) + min_lng 
            lat_predicted10 = pre_result[0][1][0] * (max_lat - min_lat) + min_lat 
            lng_predicted10 = pre_result[0][1][1] * (max_lng - min_lng) + min_lng 
            lat_predicted15 = pre_result[0][2][0] * (max_lat - min_lat) + min_lat 
            lng_predicted15 = pre_result[0][2][1] * (max_lng - min_lng) + min_lng
            #print(' predicted long:' , lng_predicted15)
            
            
            
            Distance5  = calculateDist(lat_real_short5,lng_real_short5,lat_predicted5,lng_predicted5)
            Distance10 = calculateDist(lat_real_short10,lng_real_short10,lat_predicted10,lng_predicted10)
            Distance15 = calculateDist(lat_real_short15,lng_real_short15,lat_predicted15,lng_predicted15)
            ##print('the distance:' , Distance15)
            
         
            
             
            
            lock.acquire()
            try:
                log_result = open(fileNameResult , 'a+')
                log_result.write(" %s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n" %(nodeId,timeTest,  
                                                                                     lat_real_short5, lng_real_short5,  lat_predicted5, lng_predicted5,  Distance5,
                                                                                     lat_real_short10,lng_real_short10, lat_predicted10,lng_predicted10, Distance10,
                                                                                     lat_real_short15,lng_real_short15, lat_predicted15,lng_predicted15, Distance15 ))                                                                    
                log_result.close()
            finally:
                lock.release() 
  