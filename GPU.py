import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import tensorflow as tf


#Checking GPU compatibility
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
