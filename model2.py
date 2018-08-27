import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

BATCH_SIZE = 10

samples = []

data_path = '/opt/data/pkg1/'

sample_files = ['provided_data.csv', 'turns.csv', 'recovery.csv', 'track2.csv', 'track2_rev.csv', 'cclk2.csv']
# sample_files=['driving_log.csv']
def append_lines_from(folder_path, line_array):
    with open(folder_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line_array.append(line)

for sample_file in sample_files:
    append_lines_from(data_path + sample_file, samples)

print(len(samples))
            
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))
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
                name = data_path+ 'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = center_image[55:135, :]      # Crop image here instead of in the network
                center_angle = float(batch_sample[3])
#                 flipped_image = np.fliplr(center_image)
                images.append(center_image)
#                 images.append(flipped_image)
                
                angles.append(center_angle)
#                 angles.append(-center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 80, 320  # Trimmed image format


# def append_images_and_measurements(folder, line_array, imarray, marray):
#     for line in line_array:
#         center_image = cv2.imread(folder +'IMG/' + line[0].split('/')[-1])
#         left_image = cv2.imread(folder +'IMG/' + line[1].split('/')[-1])
#         right_image = cv2.imread(folder +'IMG/' + line[2].split('/')[-1])
#         imarray.append(center_image)
#         imarray.append(left_image)
#         imarray.append(right_image)
#         steering_center = float(line[3])
#         correction = 0.15 # TUNE THIS
#         steering_left = steering_center + correction
#         steering_right = steering_center - correction
#         marray.append(steering_center)
#         marray.append(steering_left)
#         marray.append(steering_right)
# #         marray.append(measurement)
# #         flipped_image = np.fliplr(image)
# #         flipped_measurement = -measurement
# #         imarray.append(flipped_image)
# #         marray.append(flipped_measurement)
# # append_images_and_measurements('./comp_data/', lines1, images, measurements)
# # append_images_and_measurements('./test_data/', lines2, images, measurements)
# X_train = np.array(images)
# y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

model = Sequential()
# model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(row, col, ch)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
model.add(Dropout(0.4))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
# model.add(Dropout(0.4))
# model.add(Convolution2D(64,5,5, activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
# model.add(Dropout(0.4))

model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.4))

model.add(Convolution2D(96,3,3, activation='elu'))
# model.add(Dropout(0.5))
# model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(100, activation='elu'))

model.add(Dense(50, activation='elu'))
model.add(Dense(10))
model.add(Dense(1))
adam = Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# model.fit_generator(train_generator, samples_per_epoch= 2*len(train_samples), validation_data=validation_generator, \
#             nb_val_samples=2*len(validation_samples), nb_epoch=3)

# model.save('model.3.h5')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=10, callbacks=callbacks_list, verbose=1)

model.save('model.10.h5')
        
