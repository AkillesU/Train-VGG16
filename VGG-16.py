import os
import cv2
import imghdr


data_dir = 'sample'

image_exts = ['jpeg', 'jpg', 'bmp', 'png', 'jfif']

#Checking and deleting images with incompatible format
for image_class in os.listdir(data_dir): #For image class found in the sample directory
    for image in os.listdir(os.path.join(data_dir, image_class)): #For each image found in each image class
        image_path = os.path.join(data_dir, image_class, image) #Creating image path variable
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Non-compatible format" .format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Issue with image {}" .format(image_path))

#Importing tf packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import tensorflow_datasets as tfds
#GPU
physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#Creating model
model = VGG16()
model.summary()

#Creating dataset
(ds_test), ds_info = tfds.load(
    "testdata",
    as_supervised=True, #returns tuple (img, label)
    with_info=True
)
print(ds_info)
