'''
Notice:  we use this file if we want to consider only recent trajectories
'''


from splittingData import *
from numpy import hstack, array

# "granularity" means how many samples we have in each 5 seconds, we can get it based on the granularity of our system
# e.g. if we have samples every 100 milliseconds , so number of samples in 5 secs is 50

# "lag" defines the gap between samples.e.g if granulariy id 100 milliseconds, and the distance betwee samples is 5 seconds. then we want to have 
# records every 300 miliseconds, so precision here is 3  (50,3) (17,1)
# we can change lag to decrease, or increse size of dataset

# "latNum" is the column number of normalized latitude in our dataset,
# "lngNum" is the column number of normalized longitude in our dataset 
# "minSize" : how big do you want to select the size of your dataset, to be able to tarin it. 1200 , or 400( it depends on the granularity) 
# "timeClm" : the title that has been defined for the column time, it must be in string
def defineValidationset(dataset,n_step_in,n_step_out,n_features, minTime, maxTime, granularity, lag, latNum, lngNum,timeClm, minSize):

    X,y= list(), list()
    df1 = dataset[(dataset[timeClm] < minTime) | (dataset[timeClm] > maxTime)]   #minTime =50400 , maxTime = 61200
    df2 = dataset[(dataset[timeClm] >= minTime) & (dataset[timeClm] <= maxTime)]
    
    if (len(df2) > minSize):
        for step in range(0,granularity,lag):
        
             #....create train set of one car,put all information of that specific car in list............................................................................................
            df_lat2 = (df2.iloc[step::granularity,latNum]) 
            df_lng2 = (df2.iloc[step::granularity,lngNum]) 
            # define input sequence
            df_lat2 = array(df_lat2)
            df_lng2 = array(df_lng2)
            
            # convert to [rows, columns] structure
            df_lat2 = df_lat2.reshape((len(df_lat2),1))
            df_lng2 = df_lng2.reshape((len(df_lng2),1))
            # horizontally stack columns
            df_oneCar = hstack((df_lat2, df_lng2))
            split_sequences(df_oneCar,n_step_in, n_step_out,n_features,X,y) #convert time series to supervised
    else:
           # I have to check time 
        for step in range(0,granularity,lag): 
        
            #....create train set of one car,put all information of that specific car in list............................................................................................
            df_lat1 = (df1.iloc[step::granularity,latNum])
            df_lng1 = (df1.iloc[step::granularity,lngNum])  
        
            # define input sequence
            df_lat1 = array(df_lat1)
            df_lng1 = array(df_lng1)
           
            # convert to [rows, columns] structure
            df_lat1 = df_lat1.reshape((len(df_lat1),1))
            df_lng1 = df_lng1.reshape((len(df_lng1),1))
            # horizontally stack columns
            df_oneCar = hstack((df_lat1, df_lng1))
            split_sequences(df_oneCar,n_step_in, n_step_out,n_features,X,y) #convert time series to supervised
        
                
    return array(X),array(y)