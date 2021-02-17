import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

#getting mnist dataset from tensorflow.keras.datasets
data = k.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()


#normalize the features data since it ranges from 0 - 255
#making it in the range of 0 to 1
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train = k.utils.normalize(x_train, axis=1)
x_test = k.utils.normalize(x_test, axis=1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#build our model
model = Sequential()
model.add(Conv2D(64, 3, activation = "relu", input_shape = x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3),activation = "relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(10, activation = "softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

#training the model
history = model.fit(x_train, y_train , batch_size = 32, epochs = 3)

#visualizing the accuracy and loss
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss'], loc='upper left')
plt.show()

#visualizing misclassifications
loss, accuracy = model.evaluate(x_test, y_test)
score = model.predict(x_test)
err = y_test - score
highest_err_index = np.argmax(err, axis = 0)
miss = x_test[highest_err_index]
pred = score[highest_err_index]
true = y_test[highest_err_index]
for i in range (0,5):
    plt.title("classified as "
    + np.argmax(pred[i]).astype(str) 
    +" true value = " + np.argmax(true[i]).astype(str))
    plt.imshow(miss[i],cmap = plt.cm.binary)
    plt.show()




