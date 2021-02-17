import glob
import random
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as k
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

#number of samples used from the whole data set and the test dataset ratio
samples = 10000
test_ratio = 0.25

#dimensions of the images
w = 64
h = 64

#getting the paths to all the images in the respective folders
dogs = glob.glob("PetImages/Dog/*")
cats = glob.glob("PetImages/Cat/*")

#randomly selecting sample datasets from the whole dataset
dog_pics = random.sample(dogs, samples)
cat_pics = random.sample(cats, samples)

#reading the dog datasets using opencv and storing them in an array
#using try and except for the errors that arise from the opencv
#when it encounters damaged or empty data
dog_data = []
for path in dog_pics:
    try:
        image_data = cv2.imread(path)
        resized_image = cv2.resize(image_data, (w,h))
    except Exception as e:
        pass
    dog_data.append(resized_image)
dog_data = np.array(dog_data)

#doing the same for the cat dataset
cat_data = []
for path in cat_pics:
    try:
        image_data = cv2.imread(path)
        resized_image = cv2.resize(image_data, (w,h))
    except Exception as e:
        pass
    cat_data.append(resized_image)
cat_data = np.array(cat_data)

#concatenating the dog and cat datasets into one dataset to be used for training and testing
all_data = np.concatenate((dog_data, cat_data), axis = 0)
all_labels = np.concatenate(
    (np.ones(len(dog_data,)),
    np.zeros(len(cat_data),))
    , axis = 0).astype(np.int)

#normalizing the data
all_data = all_data.astype(np.float)/255.0

#splitting the dataset into training and testing.
x_train, x_test, y_train, y_test = train_test_split(
    all_data,
    all_labels,
    test_size = test_ratio)

#building our CNN model
model = Sequential()
model.add(Conv2D(64, 3, activation = "relu", input_shape = (h,w,3),data_format="channels_last"))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3),activation = "relu", data_format="channels_last"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(2, activation = "softmax"))

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
#training the model 
history = model.fit(x_train, y_train , batch_size = 32, epochs = 10)

#visualizing the accuracy and loss
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss'], loc='upper left')
plt.show()
#evaluating the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)
score = model.predict(x_test)
f_test = to_categorical(y_test)
err = f_test - score
highest_err_index = np.argmax(err, axis = 0)
miss = x_test[highest_err_index]
pred = score[highest_err_index]
true = f_test[highest_err_index]
for i in range (0,5):
    if i<len(highest_err_index):
        plt.title("classified as "
        + np.argmax(pred[i]).astype(str) 
        +" true value = " + np.argmax(true[i]).astype(str))
        plt.imshow(miss[i],cmap = plt.cm.binary)
        plt.show()
        

