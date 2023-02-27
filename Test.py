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

#Initializing wandb
wandb.init(project="Train-VGG16", entity="a-rechardt", config={"epochs":epochs, "batch_size":batch_size, "learning_rate":learning_rate})

#assign directory
directory="/fast-data22/datasets/ILSVRC/2012/clsloc/val_white"

dataset = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels = 'inferred',
    label_mode= "int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224,224),
    shuffle=True,#Shuffles data to create "random" dataset from directory
    seed=123
)


inputs = keras.Input(shape=[224,224,3], batch_size= batch_size)
x = tf.keras.applications.vgg16.preprocess_input(inputs)
model = tf.keras.applications.VGG16(weights="imagenet")
outputs = model(x)
model = keras.Model(inputs, outputs)

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

results = model.evaluate(dataset, batch_size=32, callbacks=WandbCallback())

print(results)
wandb.finish()
exit("Done")