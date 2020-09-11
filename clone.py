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
cf = 0.2 #correction factor for left and right images

#remember to update this for loop, looks too clumsy
for line in lines:
    center_path = line[0]
    left_path = line[1]
    right_path = line[2]
    center_image = cv2.imread(center_path)
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)
    center_image_flipped = np.fliplr(center_image)
    left_image_flipped = np.fliplr(left_image)
    right_image_flipped = np.fliplr(right_image)
    imgs = [center_image, left_image, right_image, center_image_flipped, left_image_flipped, right_image_flipped]
    images.extend(imgs)
    measurement = float(line[3])
    angs= [measurement, measurement+cf,measurement-cf, -measurement, -(measurement+cf), -(measurement-cf)]
    measurements.extend(angs)

"""
with open('C:/Users/manas/OneDrive - Clemson University/Documents/SDC_github/driving_log.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "C:/Users/manas/OneDrive - Clemson University/Documents/SDC_github/IMG" # fill in the path to your training IMG directory
        img_center = process_image(np.asarray(Image.open(path + row[0])))
        img_left = process_image(np.asarray(Image.open(path + row[1])))
        img_right = process_image(np.asarray(Image.open(path + row[2])))

        # add images and angles to data set
        car_images.extend(img_center, img_left, img_right)
        steering_angles.extend(steering_center, steering_left, steering_right)
"""

X_train = np.array(images)
Y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(tf.keras.layers.Conv2D(6, 5, (2,4), padding="valid", activation='relu'))
model.add(tf.keras.layers.Conv2D(12, 5, (2,2), padding="valid", activation='relu'))
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

