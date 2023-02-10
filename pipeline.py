import keras.losses
import tensorflow as tf
import os
import tensorflow_datasets as tfds

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_ds = tf.keras.utils.image_dataset_from_directory(
    "C:/Users/arech/Documents/Imagenet/imagenet-a/",
    labels = 'inferred',
    label_mode= "int",
    color_mode="rgb",
    batch_size=8,
    image_size=(224,224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training"
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    "C:/Users/arech/Documents/Imagenet/imagenet-a/",
    labels='inferred',
    label_mode="int",
    color_mode="rgb",
    batch_size=8,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation"
)

for images, labels in train_ds.take(1):
    print(images.shape)
    print(labels.shape)
model = tf.keras.applications.VGG16()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.fit(train_ds, epochs=2, verbose=2)




