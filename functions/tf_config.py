import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow import keras
tf.debugging.set_log_device_placement(True)

def config_gpu(gpu_index, loud=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            
            if loud:
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
                print(f"Configured to run on GPU {gpu_index}")
                
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
