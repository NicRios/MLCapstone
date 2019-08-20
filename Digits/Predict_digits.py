import tensorflow as tf
import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils

word = ""
num_classes = 10;


model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=(16, 16, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(units= num_classes, activation='softmax'))
model.load_weights('weights3.model')


# test_image = cv2.imread('test_images/test13.png', 0)
# test_image = cv2.resize(test_image, (16,16))
# test_image = (test_image * 1./255)
# rows = test_image.shape[0]
# cols = test_image.shape[1]
# for x in range(0, rows):
#     for y in range(0, cols):
#         if test_image[x][y] < 0.45:
#             test_image[x][y] = 1.0
#         else:
#             test_image[x][y] = 0.0
#
# print(test_image)
# test_image = test_image.reshape(1,16,16,1)
# # print(test_image)
#
# prediction = model.predict_classes(test_image)
#
# print(prediction)

def letter(index_val):
    word = ""
    if index_val == 0:
        return word+"0"
    elif index_val == 1:
        return word+"1"
    elif index_val == 2:
        return word+"2"
    elif index_val == 3:
        return word+"3"
    elif index_val == 4:
        return word+"4"
    elif index_val == 5:
        return word+"5"
    elif index_val == 6:
        return word+"6"
    elif index_val == 7:
        return word+"7"
    elif index_val == 8:
        return word+"8"
    elif index_val == 9:
        return word+"9"
    else:
        return
        ("Could not determine letter")

for image in os.listdir('Result'):
    image_path = os.path.join('Result',image)
    test_image = cv2.imread(image_path, 0)
    test_image = cv2.resize(test_image, (16,16))
    test_image = (test_image * 1./255)
    rows = test_image.shape[0]
    cols = test_image.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if test_image[x][y] < 0.45:
                test_image[x][y] = 1.0
            else:
                test_image[x][y] = 0.0
    print(test_image)
    test_image = np.reshape(test_image, (1,16,16,1))
    prediction = model.predict(test_image)
    index = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape)
    index_val = index[1]
    # print(index_val)
    # index_letter_array = np.asarray(index)
    # index_val = index_letter_array[0]
    # print(index_val)
    word += letter(index_val)



print(word)
open('text_file.txt', 'w').close()
f = open("text_file.txt","a+")
f.write(word)
folder = 'Result/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)
