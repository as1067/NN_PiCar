import cv2
from keras import Model,Sequential
import keras.layers as l
import csv
import numpy as np
import keras
from random import sample
from keras.optimizers import Adam
import sys

# Data preprocessing
steering = []
images = []
cells = []
for i in range(2,10):
    vidcap = cv2.VideoCapture("models/edge"+str(i)+".avi")
    print("preparing data")
    count = 0
    # i = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            # print(image)
            image = cv2.resize(image,(80,60))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image,2)
            # print(image.shape)
            image = np.true_divide(image,255)
            images.append(image)

    with open("output"+str(i)+".csv") as steer:
        reader = csv.reader(steer,delimiter=",")
        for row in reader:
                steering.append(int(row[2]))

images = np.asarray(images)
steering = np.asarray(steering)
# x = images
# y = steering
ids = []
for i in range(images.shape[0]):
    ids.append(i)
ids = sample(ids,images.shape[0])
# print(ids)
# sys.exit()
x = []
y=[]
for i in ids:
    x.append(images[i])
    y.append(steering[i])
x = np.asarray(x)
y = np.asarray(y)
y = np.divide(y,200)

#Neural Network Setup
batch_size = 50
dropout = .4
model = Sequential()
model.add(l.Conv2D(256,activation="relu",kernel_size=(3,3),input_shape=(60,80,1),data_format="channels_last"))
model.add(l.Conv2D(128,activation="relu",kernel_size=(3,3),data_format="channels_last"))
model.add(l.Conv2D(64,activation="relu",kernel_size=(3,3),data_format="channels_last"))
model.add(l.Flatten())
# model.add(l.Reshape((60,80)))
# model.add(l.SimpleRNN(100,activation="relu",dropout=dropout,recurrent_dropout=dropout))
model.add(l.Dense(200,activation="sigmoid"))
model.add(l.Dropout(dropout))
model.add(l.BatchNormalization())
model.add(l.Dense(100,activation="sigmoid"))
model.add(l.Dropout(dropout))
model.add(l.BatchNormalization())
model.add(l.Dense(50,activation="sigmoid"))
model.add(l.Dropout(dropout))
model.add(l.BatchNormalization())
model.add(l.Dense(1,activation="relu"))
model.compile(optimizer=Adam(),loss="mean_squared_error")
# model = keras.models.load_model("checkpoint/model_0.h5")

#Neural Network Training
print("starting training")
model.fit(x,y,batch_size=batch_size,epochs=5,validation_split=.3)
model.save("checkpoint/model_11.h5")


