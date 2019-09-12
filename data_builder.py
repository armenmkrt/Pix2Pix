import cv2
import os
from face_labeling import target_builder


dataset_path = '/home/ml-05/Downloads/nikol_live2.mp4'
target_path = '/home/ml-05/Documents/Vid2Vid/nikol_dataset/final_target'
input_path = '/home/ml-05/Documents/Vid2Vid/nikol_dataset/final_input'


cap = cv2.VideoCapture(dataset_path)
i = 23426


if cap.isOpened() == False:
    print('Error opening video stream or file')
while cap.isOpened():
    ret, frame = cap.read()
    try:
        im_edges, croped_image = target_builder(frame, (640, 360))
        input_image = cv2.resize(im_edges, (360, 360), interpolation=cv2.INTER_CUBIC)
        target_image = cv2.resize(croped_image, (360, 360), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(target_path + '/final_target' + str(i) + '.jpg', target_image)
        cv2.imwrite(input_path + '/final_input' + str(i) + '.jpg', input_image)
        i += 1
    except TypeError:
        print("Face not detected!!!")
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    print(i)


cap.release()
cv2.destroyAllWindows()
