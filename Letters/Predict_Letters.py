import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dropout
from tqdm import tqdm
import operator


word = ""

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('weights3.model')

def letter(index_val):
    word = ""
    if index_val == 0:
        return word+"A"
    elif index_val == 1:
        return word+"B"
    elif index_val == 2:
        return word+"C"
    elif index_val == 3:
        return word+"D"
    elif index_val == 4:
        return word+"E"
    elif index_val == 5:
        return word+"F"
    elif index_val == 6:
        return word+"G"
    elif index_val == 7:
        return word+"H"
    elif index_val == 8:
        return word+"I"
    elif index_val == 9:
        return word+"J"
    elif index_val == 10:
        return word+"K"
    elif index_val == 11:
        return word+"L"
    elif index_val == 12:
        return word+"M"
    elif index_val == 13:
        return word+"N"
    elif index_val == 14:
        return word+"O"
    elif index_val == 15:
        return word+"P"
    elif index_val == 16:
        return word+"Q"
    elif index_val == 17:
        return word+"R"
    elif index_val == 18:
        return word+"S"
    elif index_val == 19:
        return word+"T"
    elif index_val == 20:
        return word+"U"
    elif index_val ==21:
        return word+"V"
    elif index_val == 22:
        return word+"W"
    elif index_val == 23:
        return word+"X"
    elif index_val == 24:
        return word+"Y"
    elif index_val == 25:
        return word+"Z"
    else:
        return
        ("Could not determine letter")

    # print(word)

for image in os.listdir('Result'):
    image_path = os.path.join('Result',image)
    test_image = cv2.imread(image_path, 0)
    test_image = cv2.resize(test_image, (28,28))
    test_image = (test_image * 1./255)
    rows = test_image.shape[0]
    cols = test_image.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if test_image[x][y] < 0.3:
                test_image[x][y] = 1.0
            else:
                test_image[x][y] = 0.0
    print(test_image)
    test_image = np.reshape(test_image, (1,28,28,1))
    # print(test_image)
    prediction = model.predict(test_image, steps = 1)
    index = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape)
    index_letter_array = np.asarray(index)
    index_val = index_letter_array[1]
    word += letter(index_val)
    print(prediction)

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
