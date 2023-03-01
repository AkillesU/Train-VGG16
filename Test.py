import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras.losses
import tensorflow as tf
#import tensorflow_datasets as tfds
import wandb
from wandb.keras import WandbCallback


#Checking GPU compatibility
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

epochs = 10
batch_size = 32
learning_rate = 0.001
weight_decay = 0.0005
momentum = 0.9

#Creating model from keras library: pretrained vgg16 model
inputs = keras.Input(shape=[224,224,3], batch_size= batch_size)
x = tf.keras.applications.vgg16.preprocess_input(inputs)
model = tf.keras.applications.VGG16(weights="imagenet")
model.summary()
model.layers[-1].trainable = False #Freeze all weights
outputs = model(x)
model = keras.Model(inputs,outputs)

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate= learning_rate, weight_decay=weight_decay, use_ema=True, ema_momentum=momentum), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.layers[-1].layers[1]
