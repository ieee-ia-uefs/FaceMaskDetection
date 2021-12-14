import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers 
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


data_dir = 'data'

batch_size = 32
img_height = 224
img_width = 224

def load_images():

  dataGen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                            rescale=1./255,
                                                            rotation_range=20,
                                                            zoom_range=0.15,
                                                            width_shift_range=0.2,
                                                            height_shift_range=0.2,
                                                            shear_range=0.15,
                                                            validation_split=0.2,
                                                            fill_mode='nearest',
                                                            horizontal_flip=True
                                                          )

  train = dataGen.flow_from_directory(
                                    directory=data_dir,
                                    batch_size=batch_size,
                                    target_size=(img_height, img_width), 
                                    shuffle=True,
                                    subset='training'
                                  )

  validation = dataGen.flow_from_directory(
                                    directory=data_dir,
                                    batch_size=batch_size,
                                    target_size=(img_height, img_width), 
                                    shuffle=True,
                                    subset='validation'
                                  )

  return train, validation

train, validation = load_images()
num_classes = train.num_classes

base_model = MobileNetV2(input_shape=(img_height, img_width, 3), weights='imagenet',  include_top=False)
base_model.trainable = False

model = Sequential([
                    base_model,
                    layers.AveragePooling2D(pool_size=(7,7)),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics='accuracy')

model.fit(train, validation_data = validation, epochs=5)
model.save("facemask_detector.model", save_format="h5")