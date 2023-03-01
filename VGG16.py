import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import tensorflow as tf
import keras.losses
import wandb
from wandb.keras import WandbCallback

#Checking GPU compatibility
gpus = tf.config.list_physical_devices('GPU')
print(gpus)


epochs = 4
batch_size = 32
learning_rate = 0.0001
weight_decay = 0.0005
momentum = 0.9
#Initializing wandb
wandb.init(project="Train-VGG16", entity="a-rechardt", config={"epochs":epochs, "batch_size":batch_size, "learning_rate":learning_rate})



#Creating training dataset from fast-22 imagenet directory, defining batch size and prerpocessing image size
train_ds = tf.keras.utils.image_dataset_from_directory(
    "/fast-data22/datasets/ILSVRC/2012/clsloc/train",
    labels = 'inferred',
    label_mode= "int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224,224),
    shuffle=True,#Shuffles data to create "random" dataset from directory
    seed=123,
    validation_split=0.2,
    subset= "training"
)
#Creating validation dataset from fast-22 imagenet directory, defining batch size and prerpocessing image size
validation_ds = tf.keras.utils.image_dataset_from_directory(
    "/fast-data22/datasets/ILSVRC/2012/clsloc/train",
    labels='inferred',
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True, #Shuffles data to create "random" dataset from directory
    seed=123,
    validation_split=0.2,
    subset= "validation"
)
#Checking the image and label object shapes
for images, labels in train_ds.take(1):
    print(images.shape)
    print(labels.shape)

#Creating model from keras library: pretrained vgg16 model
inputs = keras.Input(shape=[224,224,3], batch_size= batch_size)
x = tf.keras.applications.vgg16.preprocess_input(inputs)
model = tf.keras.applications.VGG16(weights="imagenet")
model = model.trainable = False #Freeze all weights
outputs = model(x)
model = keras.Model(inputs, outputs)

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
#Training model and sending stats to wandb
model.fit(train_ds, epochs= epochs, verbose=1, validation_data=validation_ds, callbacks=[WandbCallback()])

model.save_weights('trained_weights_VGG16/')

wandb.finish()

exit("Done")


