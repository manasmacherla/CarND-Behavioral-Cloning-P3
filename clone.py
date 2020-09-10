import cv2
import csv
import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
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
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 10)

model.save('model.h5')

