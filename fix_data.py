import cv2
import csv

ids = [2]

for i in ids:
    vidcap = cv2.VideoCapture("output"+str(i)+".avi")
    success = True
    frames = 0
    while success:
        success, image = vidcap.read()
        if success:
            frames += 1
    print(frames)
    f = open("output"+str(i)+"old.csv","r")
    w = open("output"+str(i)+".csv","w")
    f.readline()
    count = 0
    for row in f:
        count+=1
    print(count)
    dif = count-frames-1
    print(dif)
    count = 0
    f = open("output"+str(i)+"old.csv","r")
    for row in f:
        if count>dif:
            # print(str(count)+'\t'+str(dif))
            w.write(row)
        count+=1

