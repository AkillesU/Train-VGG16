import os


import tensorflow as tf



epochs = 10
batch_size = 32
learning_rate = 0.0001
weight_decay = 0.0005


#Creating model from keras library: pretrained vgg16 model
inputs = tf.keras.Input(shape=(224,224,3)) #Input layer takes in arrays with "width" and "height" (any) and 3 color channels
x = tf.keras.applications.vgg16.preprocess_input(inputs) #Vgg16 preprocessing layer takes in arrays (224,224,3) and preprocesses: (scales, rgb to bgr etc.)
model = tf.keras.applications.VGG16(weights="imagenet") #Loading vgg-16 model with pretrained weights
model.summary()
#model.trainable = False #Freeze all weights
outputs = model(x)
model = tf.keras.Model(inputs,outputs)


print(model.layers[3].layers[1])