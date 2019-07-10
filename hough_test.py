import cv2
import numpy as np
cfg_cam_res = (320, 240)
cfg_cam_fps = 30

i = 2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vidfile = cv2.VideoWriter("models/lines" + str(i) + ".avi", fourcc, cfg_cam_fps, cfg_cam_res, 0)
vidcap = cv2.VideoCapture("output" + str(i) + ".avi")

def get_lines(image):
    v = np.median(image)
    l = np.zeros((240,320))
    sigma = .33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper, apertureSize=3)
    minLineLength = 30
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            cv2.polylines(l, [pts], True, (255))
    return l


success = True
count = 0
while success:
    success, image = vidcap.read()
    if success:
        l = get_lines(image)
        l = np.uint8(l)
        # print(e.shape)
        vidfile.write(l)
        print(count)
        count+=1
