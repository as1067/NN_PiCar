import numpy as np
import argparse
import glob
import cv2
cfg_cam_res = (80, 60)
cfg_cam_fps = 30
# from scipy import ndimage as ndi
# from skimage import feature

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper,apertureSize=3)
    return edged

for i in range(2,10):
    print(i)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidfile = cv2.VideoWriter("models/edge"+str(i)+".avi", fourcc, cfg_cam_fps, cfg_cam_res,0)
    vidcap = cv2.VideoCapture("output" + str(i) + ".avi")
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            image = cv2.resize(image,(80,60))
            e = auto_canny(image)
            # print(e.shape)
            vidfile.write(e)
