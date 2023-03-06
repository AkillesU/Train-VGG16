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


epochs = 10
batch_size = 32
learning_rate = 0.00001
weight_decay = 0.0005
momentum = 0.9
#Initializing wandb
wandb.init(project="Train-VGG16", entity="a-rechardt", config={"epochs":epochs, "batch_size":batch_size, "learning_rate":learning_rate, "momentum":momentum, "weight_decay":weight_decay})

#defining checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="final_train{batch:02d}epoch{epoch:02d}", save_weights_only=False, save_freq=4000, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="auto", patience=1, verbose=1)
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

model = tf.keras.models.load_model("Models/full_train32000epoch01")

for layer in range(0,21):
    model.layers[3].layers[layer].trainable = False

#Setting model training hyperparameters
model.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate= learning_rate, weight_decay=weight_decay, use_ema=True, ema_momentum=momentum), #Change to AdamW and add momentum and decay
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

initial_test = model.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
print(initial_test)
#Training model and sending stats to wandb
model.fit(train_ds, epochs=epochs, verbose=1, validation_data=validation_ds, callbacks=[WandbCallback(),cp_callback,earlystopping])

final_test = model.evaluate(test_ds, batch_size=batch_size, callbacks=[WandbCallback()], verbose=1)
print(final_test)

wandb.finish()

exit("Done")