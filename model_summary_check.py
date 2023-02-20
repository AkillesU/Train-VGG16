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


epochs = 10
batch_size = 32
learning_rate = 0.0001
weight_decay = 0.0005


#Creating model from keras library: pretrained vgg16 model
model = tf.keras.applications.VGG16(weights='imagenet')

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate, weight_decay= weight_decay),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
#Checking model architecture
model.summary()