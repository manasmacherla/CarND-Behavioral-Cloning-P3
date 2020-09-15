import cv2
import csv
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Flatten, Lambda, Cropping2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MSE

from tensorflow.keras.applications.xception import Xception

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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

model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

main_input = Input(shape=(160,320,3))
resized_input = Lambda(lambda image: tf.image.resize(image, (299, 299)))(main_input)

model = model(resized_input)

flattened1 = Flatten()(model)
dense1 = Dense(120, activation = 'relu')(flattened1)
predictions = Dense(1, activation = 'relu')(dense1)

model = Model(inputs=main_input, outputs=predictions)
model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, epochs = 5)
model.save('model.h5')

"""
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
#model.add(tf.keras.layers.Conv2D(6, 5, (2,4), padding="valid", activation='relu'))
model.add(tf.keras.layers.Conv2D(12, 5, (2,2), padding="valid", activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(12, 5, 1, padding="valid", activation='relu')) 
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.MaxPooling2D(2,2,padding='valid')) 
model.add(tf.keras.layers.Conv2D(16, 5, 1, padding="valid", activation='relu')) 
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.MaxPooling2D(2,2,padding='valid')) 
model.add(Flatten()) 
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(1))
"""

#trying to use pretrained model to see how it works



"""
model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, epochs = 5)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""

#things that can be done to further improve the model
#adding dropout to reduce overfitting 
#using generators
#better preprocessing - YCbCr format/ other image formats
#transfer learning using pretrained weights ResNet, VGG16 and Nvidia end to end pipeline
#make a graph of the loss and accuracy of all these models
#advanced challenge track 

