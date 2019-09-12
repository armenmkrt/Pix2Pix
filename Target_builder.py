import cv2
import os
from face_labeling import target_builder
from imutils import face_utils
import imutils
import numpy as np
import sys
from skimage import io


dataset_path = '/home/armen/Downloads/nikol_live_video'
target_path = '/home/armen/Desktop/Vid2Vid/nikol_dataset/target'
input_path = '/home/armen/Desktop/Vid2Vid/nikol_dataset/input'


cap = cv2.VideoCapture(dataset_path)
i = 1
if cap.isOpened() == False:
    print('Error opening video stream or file')
while cap.isOpened():
    ret, frame = cap.read()
    if i >= 23450:
        im_edges, croped_image = target_builder(frame, (640, 360))
        cv2.imwrite(os.path.join(target_path + '/target' + str(i) + '.jpg'), croped_image)
        cv2.imwrite(os.path.join(input_path + '/input' + str(i) + '.jpg'), im_edges)
    # cv2.imshow('Image edges', im_edges)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    print(i)
    i += 1
cap.release()
cv2.destroyAllWindows()
