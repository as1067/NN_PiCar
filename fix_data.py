import cv2
import csv

ids = [4,7,9]

for i in ids:
    vidcap = cv2.VideoCapture("output"+str(i)+".avi")
    success = True
    frames = 0
    while success:
        success, image = vidcap.read()
        if success:
            frames += 1
    print(frames)
    f = open("output"+str(i)+".csv","r")
    w = open("1output"+str(i)+".csv","w")
    f.readline()
    for i in range(frames):
        w.write(f.readline())


