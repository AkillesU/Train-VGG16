import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import keras.losses
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

#Checking GPU compatibility
gpus = tf.config.list_physical_devices('GPU')
print(gpus)