import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import scipy
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
learning_rate = 0.0001
weight_decay = 0.0005
momentum = 0.9
#directory = "/fast-data22/datasets/ILSVRC/2012/clsloc/train"
directory = "C:/Users/arech/Documents/Imagenet/imagenet-a"
#Initializing wandb
#wandb.init(project="Train-VGG16", entity="a-rechardt", config={"epochs":epochs, "batch_size":batch_size, "learning_rate":learning_rate})

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, horizontal_flip= True, channel_shift_range=0.2) #Creating imagegenerator

train_gen = img_gen.flow_from_directory(directory=directory, #creating images and labels for training ds
                                                  keep_aspect_ratio=True,
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  color_mode="rgb",
                                                  class_mode="sparse", #labels int
                                                  shuffle=True,
                                                  seed=123,
                                                  subset= "training")

train_ds = tf.data.Dataset.from_generator(lambda: train_gen.flow_from_directory(directory), #creating training ds
                                          output_types=(tf.float32, tf.float32),
                                          output_shapes=([batch_size,224,224,3], [32,1000])) #change to 1000

val_gen = img_gen.flow_from_directory(directory=directory,#creating images and labels for validation ds
                                                  keep_aspect_ratio=True,
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  color_mode="rgb",
                                                  class_mode="sparse", #labels int
                                                  shuffle=True,
                                                  seed=123,
                                                  subset= "validation")

validation_ds = tf.data.Dataset.from_generator(lambda: val_gen.flow_from_directory(directory), #creating validation ds
                                          output_types=(tf.float32, tf.float32),
                                          output_shapes=([batch_size,224,224,3], [32 ,1000])) #change to 1000

print(train_ds)

#for images, labels in train_ds.take(1):
#  print('images.shape: ', images.shape)
#  print('labels.shape: ', labels.shape)


#Creating model from keras library: pretrained vgg16 model
vgg16 = tf.keras.applications.VGG16(weights="imagenet")


inputs = keras.Input(shape=(224,224,3)) #Input layer takes in arrays with "width" and "height" (any) and 3 color channels
x = tf.keras.applications.vgg16.preprocess_input(inputs) #Vgg16 preprocessing layer takes in arrays (224,224,3) and preprocesses: (scales, rgb to bgr etc.)

base_model = tf.keras.Sequential(
    [

        vgg16.layers[0],
        vgg16.layers[1],
        vgg16.layers[2],
        vgg16.layers[3],
        vgg16.layers[4],
        vgg16.layers[5],
        vgg16.layers[6],
        vgg16.layers[7],
        vgg16.layers[8],
        vgg16.layers[9],
        vgg16.layers[10],
        vgg16.layers[11],
        vgg16.layers[12],
        vgg16.layers[13],
        vgg16.layers[14],
        vgg16.layers[15],
        vgg16.layers[16],
        vgg16.layers[17],
        vgg16.layers[18],
        vgg16.layers[19],
        vgg16.layers[20],
        vgg16.layers[21],
        vgg16.layers[22]
    ]
)
output = base_model(x)
model = tf.keras.Model(inputs,output)




#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate= learning_rate, weight_decay=weight_decay, use_ema=True, ema_momentum=momentum), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.summary()
#Training model and sending stats to wandb
model.fit(train_ds, epochs= epochs, verbose=1, validation_data=validation_ds) #, callbacks=[WandbCallback()]

model.save_weights('trained_weights_VGG16/')

wandb.finish()

exit("Done")

model.fit_generator