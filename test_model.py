import cv2
import numpy as np
import csv
import keras
import time
i = 3
j = 7
steering = []
vidcap = cv2.VideoCapture("output" + str(i) + ".avi")
with open("output" + str(i) + ".csv") as f:
    reader = csv.reader(f,delimiter=",")
    for row in reader:
        steering.append(int(row[2]))
model = keras.models.load_model("models/model_"+str(j)+".h5")
count = 0
success = True
print("Starting")
while success:
    success, image = vidcap.read()
    if success:
        # print(image)
        start = time.time()
        image = cv2.resize(image, (80, 60))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, 2)
        image = np.true_divide(image, 255)
        images = [image]
        images = np.asarray(images)
        angles = model.predict(images,batch_size=1)
        angle = int(angles[0])
        actual = steering[count]
        end = time.time()
        dif = end-start
        print(str(angle)+"\t"+str(actual)+"\t"+str(dif))
        count+=1
