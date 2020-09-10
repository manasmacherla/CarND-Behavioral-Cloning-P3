import cv2
import csv
import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MSE

lines = []
#opening the csv file with the file names and measurements 
with open('C:/Users/manas/OneDrive - Clemson University/Documents/SDC_github/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#creating two empty lists for images and curresponding measurements
images = []
measurements = []

for line in lines:
    source_path = line[0]
    image = cv2.imread(source_path)
    images.append(image)
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(-measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(tf.keras.layers.Conv2D(6, (34,68), (2,4), padding="valid", activation='relu'))
model.add(tf.keras.layers.Conv2D(12, (2,2), (2,2), padding="valid", activation='relu'))
model.add(tf.keras.layers.Conv2D(12, 5, 1, padding="valid", activation='relu')) #shape = 28*28*12
model.add(tf.keras.layers.MaxPooling2D(2,2,padding='valid')) #shape = 14*14*12
model.add(tf.keras.layers.Conv2D(16, 5, 1, padding="valid", activation='relu')) #shape = 10*10*16
model.add(tf.keras.layers.MaxPooling2D(2,2,padding='valid')) #shape = 5*5*16
model.add(Flatten()) #shape = 400
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(1))

#shape = model.output_shape
#print(shape)

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

model.save('model.h5')

