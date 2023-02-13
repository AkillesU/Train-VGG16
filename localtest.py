import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.losses
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
#wandb.init(project="Train-VGG16", entity="a-rechardt")

epochs = 20
batch_size = 32
learning_rate = 0.001

#Creating training dataset from fast-22 imagenet directory, defining batch size and prerpocessing image size
train_ds = tf.keras.utils.image_dataset_from_directory(
    "C:/Users/arech/Documents/Imagenet/imagenet-a",
    labels = 'inferred',
    label_mode= "int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224,224),
    shuffle=True,#Shuffles data to create "random" dataset from directory
    seed=123,
    subset="training",
    validation_split= 0.2
)
#Creating validation dataset from fast-22 imagenet directory, defining batch size and prerpocessing image size
validation_ds = tf.keras.utils.image_dataset_from_directory(
    "C:/Users/arech/Documents/Imagenet/imagenet-a",
    labels='inferred',
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True, #Shuffles data to create "random" dataset from directory
    seed=123,
    subset="validation",
    validation_split= 0.2
)
#Checking the image and label object shapes
for images, labels in train_ds.take(1):
    print(images.shape)
    print(labels.shape)
#Setting hyperparameters to wandb
wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "batch_size": batch_size
}
#Creating model from keras library: pretrained vgg16 model
model = tf.keras.applications.VGG16()

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)
#Training model and sending stats to wandb
model.fit(train_ds, epochs= epochs, verbose=1, validation_data=validation_ds, callbacks=[WandbCallback()])

model.save_weights('trained_weights_VGG16/')
