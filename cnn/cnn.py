import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.preprocessing import image

# loading dataset

# generating training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# generating testing data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# creating training set
training_set = train_datagen.flow_from_directory(
    '/home/theesh/PycharmProjects/DeepLearning/cnn/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# creating test set
test_set = test_datagen.flow_from_directory(
    '/home/theesh/PycharmProjects/DeepLearning/cnn/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

cnn = Sequential()

# first convolutional layer and max pooling layer
cnn.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
               input_shape=[64, 64, 3]))
cnn.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))

# second convolutional layer and max pooling layer
cnn.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
               input_shape=[64, 64, 3]))
cnn.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))

# flatterning
cnn.add(Flatten())

# full connected layer
cnn.add(Dense(units=128, activation='relu'))

# output layer
cnn.add(Dense(units=1, activation='sigmoid'))

# training the cnn
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit_generator(
    training_set,
    steps_per_epoch=int(8000/32),
    epochs=25,
    validation_data=test_set,
    validation_steps=int(2000/32)
)

# single prediction
test_image = image.load_img(
    '/home/theesh/PycharmProjects/DeepLearning/cnn/dataset/single_prediction/cat_or_dog_1.jpg',
    target_size=(64, 64)
)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
