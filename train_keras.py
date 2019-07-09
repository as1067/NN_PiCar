import cv2
from keras import Model,Sequential
import keras.layers as l
import csv
import numpy as np
import keras
from random import sample
from keras.optimizers import Adam

# Data preprocessing
steering = []
images = []
for i in range(2,10):
    vidcap = cv2.VideoCapture("output"+str(i)+".avi")
    print("preparing data")
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

#Neural Network Setup
batch_size = 100
dropout = .4
model = Sequential()
model.add(l.Conv2D(64,activation="relu",kernel_size=(3,3),input_shape=(60,80,1),data_format="channels_last"))
model.add(l.Conv2D(32,activation="relu",kernel_size=(3,3),data_format="channels_last"))
model.add(l.Flatten())
model.add(l.Dense(50,activation="relu"))
model.add(l.Dropout(dropout))
model.add(l.BatchNormalization())
model.add(l.Dense(1,activation="relu"))
model.compile(optimizer=Adam(),loss="mean_squared_logarithmic_error")
# model = keras.models.load_model("checkpoint/model_0.h5")

#Neural Network Training
print("starting training")
model.fit(images,steering,batch_size=batch_size,epochs=20,validation_split=.3)
model.save("checkpoint/model_5.h5")


