import numpy as np
import argparse
import glob
import cv2
cfg_cam_res = (320, 240)
cfg_cam_fps = 30
# from scipy import ndimage as ndi
# from skimage import feature

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
for i in [2]:
    print(i)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidfile = cv2.VideoWriter("models/1edge"+str(i)+".avi", fourcc, cfg_cam_fps, cfg_cam_res,0)
    vidcap = cv2.VideoCapture("output" + str(i) + ".avi")
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            e = auto_canny(image)
            # print(e.shape)
            vidfile.write(e)
