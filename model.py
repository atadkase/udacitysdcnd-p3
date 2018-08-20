import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


lines1 = []
lines2 = []

def append_lines_from(folder_path, line_array):
    with open(folder_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line_array.append(line)

append_lines_from('./comp_data/driving_log.csv', lines1)
append_lines_from('./test_data/driving_log.csv', lines2)
            
# train_samples, validation_samples = train_test_split(lines, test_size=0.2)
images = []
measurements = []
# print(lines[1])
# print(lines[100])
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './comp_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def append_images_and_measurements(folder, line_array, imarray, marray):
    for line in line_array:
        current_path = folder +'IMG/' + line[0].split('/')[-1]
        image = cv2.imread(current_path)
        imarray.append(image)
        measurement = float(line[3])
        marray.append(measurement)
        flipped_image = np.fliplr(image)
        flipped_measurement = -measurement
        imarray.append(flipped_image)
        marray.append(flipped_measurement)
append_images_and_measurements('./comp_data/', lines1, images, measurements)
append_images_and_measurements('./test_data/', lines2, images, measurements)
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

model = Sequential()
model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))

model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(96,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))

model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

        