import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras.losses
import tensorflow as tf
import tensorflow-datasets as tfds
import wandb
from wandb.keras import WandbCallback

#Checking GPU compatibility
gpus = tf.config.list_physical_devices('GPU')
print(gpus)


#Creating model from keras library: pretrained vgg16 model
model = tf.keras.applications.VGG16(weights='imagenet')

test_ds = tf.keras.utils.image_dataset_from_directory(
    "/fast-data22/datasets/ILSVRC/2012/clsloc/val_white",
    labels='inferred',
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
)

test_ds = tf.keras.applications.vgg16.preprocess_input(tfds.as_numpy(test_ds))


result = model.evaluate(test_ds, verbose = 1)

print("loss", result[0])
print("accuracy",result[1])