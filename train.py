import cv2
import csv
import tensorflow as tf 

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

#print(len(images))
