import cv2
import numpy as np
import csv
import keras
import time
i = 7
j = 10
steering = []
vidcap = cv2.VideoCapture("output" + str(i) + ".avi")
with open("output" + str(i) + ".csv") as f:
    reader = csv.reader(f,delimiter=",")
    for row in reader:
        steering.append(int(row[2]))
model = keras.models.load_model("models/model_"+str(j)+".h5")
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper,apertureSize=3)
    return edged

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
        image = auto_canny(image)
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
