# classifying diseases in rice paddy plants using keras

import tensorflow as tf
import pandas as pd
import numpy as np
import os

LEARNING_RATE = 0.001
BATCH_SIZE = 16
IMG_DIM = 256
EPOCH_NUM = 15

# setting up data and labels
train_dir = './paddy-disease-classification/train_images/'
test_dir = './paddy-disease-classification/test_images/'
train_labels = os.listdir('./paddy-disease-classification/train_images')

# generating image data from the directory
img_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.1,
    rotation_range=5,
)

# actual training set
train_gen = img_data_gen.flow_from_directory(
    train_dir,
    subset="training",
    seed=42,  # cool number!
    target_size=(IMG_DIM, IMG_DIM),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# to validate parameters/etc., not actually trained on
validation_gen = img_data_gen.flow_from_directory(
    train_dir,
    subset="validation",
    seed=42,  # cool number!
    target_size=(IMG_DIM, IMG_DIM),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# test set (from Kaggle online guide)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    directory=test_dir,
    target_size=(IMG_DIM, IMG_DIM),
    batch_size=BATCH_SIZE,
    # no classes because this is the test set
    classes=['.'],
    shuffle=False
)

# defining the model's layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compiling with SGD
model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
)

# actually training the model
history_model = model.fit(
    train_gen,
    validation_data=validation_gen,
    epochs=EPOCH_NUM
)

# final predictions on test set
test_set_attempt = model.predict(test_gen)

# putting into CSV (from Kaggle online guide)
output = pd.read_csv('./paddy-disease-classification/sample_submission.csv')
output['label'] = np.argmax(test_set_attempt, axis=-1)
output['label'] = output['label'].replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train_labels)
output.to_csv("submission_final.csv", index=False)
