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
directory = "/fast-data22/datasets/ILSVRC/2012/clsloc/train"

#Initializing wandb
wandb.init(project="Train-VGG16", entity="a-rechardt", config={"epochs":epochs, "batch_size":batch_size, "learning_rate":learning_rate})

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2) #Creating imagegenerator

images, labels = next(img_gen.flow_from_directory(directory=directory, #creating images and labels for training ds
                                                  keep_aspect_ratio=True,
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  color_mode="rgb",
                                                  shuffle=True,
                                                  seed=123,
                                                  subset= "training"))

train_ds = tf.data.Dataset.from_generator(lambda: img_gen.flow_from_directory(directory), #creating training ds
                                          output_types=(tf.float32, tf.float32),
                                          output_shapes=([batch_size,224,224,3], [batch_size,1000]))

images, labels = next(img_gen.flow_from_directory(directory=directory,#creating images and labels for validation ds
                                                  keep_aspect_ratio=True,
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  color_mode="rgb",
                                                  shuffle=True,
                                                  seed=123,
                                                  subset= "validation"))

validation_ds = tf.data.Dataset.from_generator(lambda: img_gen.flow_from_directory(directory), #creating validation ds
                                          output_types=(tf.float32, tf.float32),
                                          output_shapes=([batch_size,224,224,3], [batch_size,1000]))

train_ds.element_spec

for images, labels in train_ds.take(1):
  print('images.shape: ', images.shape)
  print('labels.shape: ', labels.shape)


#Creating model from keras library: pretrained vgg16 model
inputs = keras.Input(shape=(224,224,3)) #Input layer takes in arrays with "width" and "height" (any) and 3 color channels
x = tf.keras.applications.vgg16.preprocess_input(inputs) #Vgg16 preprocessing layer takes in arrays (224,224,3) and preprocesses: (scales, rgb to bgr etc.)
model = tf.keras.applications.VGG16(weights="imagenet") #Loading vgg-16 model with pretrained weights
model.summary()
#model.trainable = False #Freeze all weights
outputs = model(x)
model = keras.Model(inputs,outputs)

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate= learning_rate, weight_decay=weight_decay, use_ema=True, ema_momentum=momentum), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()
#Training model and sending stats to wandb
model.fit(train_ds, epochs= epochs, verbose=1, validation_data=validation_ds, callbacks=[WandbCallback()])

model.save_weights('trained_weights_VGG16/')

wandb.finish()

exit("Done")
