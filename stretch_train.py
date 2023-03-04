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


epochs = 1
batch_size = 32
learning_rate = 0.0001
weight_decay = 0.0005
momentum = 0.9
#Initializing wandb
wandb.init(project="Train-VGG16", entity="a-rechardt", config={"epochs":epochs, "batch_size":batch_size, "learning_rate":learning_rate, "momentum":momentum, "weight_decay":weight_decay})



#Creating training dataset from fast-22 imagenet directory, defining batch size and prerpocessing image size
train_ds = tf.keras.utils.image_dataset_from_directory(
    "/fast-data22/datasets/ILSVRC/2012/clsloc/train",
    labels = 'inferred',
    label_mode= "int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(224,224), #cropping
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
    image_size=(224,224), #cropping
    shuffle=True, #Shuffles data to create "random" dataset from directory
    seed=123,
    validation_split=0.2,
    subset= "validation"
)
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

#Checking the image and label object shapes
for images, labels in train_ds.take(1):
    print(images.shape)
    print(labels.shape)

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

for layer in range(0,21):
    model.layers[3].layers[layer].trainable = False

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate= learning_rate, weight_decay=weight_decay, use_ema=True, ema_momentum=momentum), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

model.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
#Training model and sending stats to wandb
model.fit(train_ds, epochs=epochs, verbose=1, validation_data=validation_ds, callbacks=[WandbCallback(), tf.keras.callbacks.EarlyStopping(monitor= "val_loss", patience=1, mode="auto", verbose=1)])
model.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
model.save_weights('trained_weights_VGG16/')

wandb.finish()

exit("Done")