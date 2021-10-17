from numpy import array, hstack
from splittingData import *

# "granularity" means how many samples we have in each 5 seconds, we can get it based on the granularity of our system
# e.g. if we have samples every 100 milliseconds , so number of samples in 5 secs is 50

# "latNum" is the column number of normalized latitude in our dataset,
# "lngNum" is the column number of normalized longitude in our dataset 

def define_short_dataset(dataset,n_step_in,n_step_out,n_features,granularity,latNum,lngNum):
    X,y = list(), list()
    #....create train set of one car,put all information of that specific car in list.
    df_lat1 = (dataset.iloc[::granularity,latNum]) #Scaled lat, every 5 seconds
    df_lng1 = (dataset.iloc[::granularity,lngNum])  #Scaled lng
   
    # define input sequence
    df_lat1=array(df_lat1)
    df_lng1=array(df_lng1)

    # convert to [rows, columns] structure
    df_lat1=df_lat1.reshape((len(df_lat1),1))
    df_lng1=df_lng1.reshape((len(df_lng1),1))
    
    # horizontally stack columns
    df_oneCar = hstack((df_lat1, df_lng1))
    split_sequences(df_oneCar,n_step_in, n_step_out,n_features,X,y) #convert time series to supervised
    return array(X),array(y)