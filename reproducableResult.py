import os
import tensorflow as tf
import numpy as np
import random
def reproducableFunc():
    
    seed_num= 0 
    tf.random.set_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    PYTHONHASHSEED=0
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'   
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  