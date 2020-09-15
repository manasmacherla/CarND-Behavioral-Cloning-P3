import cv2
import csv
import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Lambda, Cropping2D
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
cf = 0.25 #correction factor of the steering angle for left and right images

#remember to update this for loop, looks too clumsy
for line in lines:
    center_path = line[0]
    left_path = line[1]
    right_path = line[2]
    center_image = cv2.imread(center_path)
    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
    left_image = cv2.imread(left_path)
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
    right_image = cv2.imread(right_path)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
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
model.add(Cropping2D(cropping=((70,25), (0,0))))
#model.add(tf.keras.layers.Conv2D(6, 5, (2,4), padding="valid", activation='relu'))
model.add(tf.keras.layers.Conv2D(12, 5, (2,2), padding="valid", activation='relu'))
model.add(tf.keras.layers.Conv2D(12, 5, 1, padding="valid", activation='relu')) 
model.add(tf.keras.layers.MaxPooling2D(2,2,padding='valid')) 
model.add(tf.keras.layers.Conv2D(16, 5, 1, padding="valid", activation='relu')) 
model.add(tf.keras.layers.MaxPooling2D(2,2,padding='valid')) 
model.add(Flatten()) 
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(1))

#shape = model.output_shape
#print(shape)

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

model.save('model.h5')

#things that can be done to further improve the model
#adding dropout to reduce overfitting 
#using generators
#better preprocessing - YCbCr format/ other image formats
#transfer learning using pretrained weights ResNet, VGG16 and Nvidia end to end pipeline
#advanced challenge track 

