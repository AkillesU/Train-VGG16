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

epochs = 2
batch_size = 32
learning_rate = 0.001
weight_decay = 0.0005
momentum = 0.9

#Initializing wandb
wandb.init(project="Train-VGG16", entity="a-rechardt", config={"epochs":epochs, "batch_size":batch_size, "learning_rate":learning_rate})

#assign directory
directory="/fast-data22/datasets/ILSVRC/2012/clsloc/train"

training = tf.keras.utils.image_dataset_from_directory(
    directory,
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
validation = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels = 'inferred',
    label_mode= "int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224,224),
    shuffle=True,#Shuffles data to create "random" dataset from directory
    seed=123,
    subset="validation",
    validation_split= 0.2
)
#assign directory
testdir="/fast-data22/datasets/ILSVRC/2012/clsloc/val_white"

testing = tf.keras.utils.image_dataset_from_directory(
    testdir,
    labels = 'inferred',
    label_mode= "int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224,224),
    shuffle=True,#Shuffles data to create "random" dataset from directory
    seed=123
)


model = tf.keras.applications.VGG16(weights="imagenet")


#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate= learning_rate, weight_decay=weight_decay, use_ema=True, ema_momentum=momentum),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()
#keras.utils.plot_model(model, show_shapes=True)
model.fit(training, epochs=epochs, validation_data=validation, verbose=1, callbacks=[WandbCallback()])

results = model.evaluate(testing, batch_size=batch_size, callbacks=WandbCallback())

print(results)
wandb.finish()
exit("Done")