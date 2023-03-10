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


batch_size = 32

#Initializing wandb
wandb.init(project="Train-VGG16", entity="a-rechardt", config={"batch_size":batch_size})


#Creating test dataset from fast-22 imagenet directory, defining batch size and prerpocessing image size
test_ds = tf.keras.utils.image_dataset_from_directory(
    "/fast-data22/datasets/ILSVRC/2012/clsloc/val_white",
    labels='inferred',
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224,224), #cropping
    shuffle=True, #Shuffles data to create "random" dataset from directory
    seed=123
)
#loading pretrained keras model
model_original = tf.keras.applications.VGG16(weights="imagenet")

#Loading finetuned model from directory
model_finetuned3 = tf.keras.models.load_model(filepath="crop_train32000epoch01")

#Loading finetuned model
model_finetuned1 = tf.keras.models.load_model(filepath="crop_train20000epoch01")

#Loading finetuned model
model_finetuned2 = tf.keras.models.load_model(filepath="crop_train12000epoch01")


#creating preprocessing layers for both models
inputs = keras.Input(shape=(224,224,3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)

#creating og model with preprocessing
original_output = model_original(x)
model_original = tf.keras.Model(inputs,original_output)



#compiling finetuned model
model_finetuned1.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model_finetuned2.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
model_finetuned3.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

#compiling original model
model_original.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

#testing original model
original_results = model_original.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
print(original_results)

finetuned_results2 = model_finetuned2.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
print(finetuned_results2)

#testing finetuned model
finetuned_results = model_finetuned1.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
print(finetuned_results)
#testing finetuned model
finetuned_results3 = model_finetuned3.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
print(finetuned_results3)

wandb.finish()
exit("Done")