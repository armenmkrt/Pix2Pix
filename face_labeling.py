import cv2
import dlib
import os
import numpy as np
from scipy.optimize import curve_fit
import warnings


MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
SIZE = (789, 444)


def face_keypoints_detector(image, model_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)

    if len(dets) > 0:
        shape = predictor(image, dets[0])
        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            points[b, 0] = shape.part(b).x
            points[b, 1] = shape.part(b).y

        return points


def read_keypoints(keypoints, size):
    # mapping from keypoints to face part
    part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]],  # face
                 [range(17, 22)],  # right eyebrow
                 [range(22, 27)],  # left eyebrow
                 [[28, 31], range(31, 36), [35, 28]],  # nose
                 [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                 [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                 [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
                 [range(60, 65), [64, 65, 66, 67, 60]]  # tongue
                 ]
    label_list = [1, 2, 2, 3, 4, 4, 5, 6]  # labeling for different facial parts

    pts = keypoints[:17, :].astype(np.int32)
    baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
    upper_pts = pts[1:-1, :].copy()
    upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
    keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

    w, h = size
    part_labels = np.zeros((h, w), np.uint8)
    for p, edge_list in enumerate(part_list):
        indices = [item for sublist in edge_list for item in sublist]
        pts = keypoints[indices, :].astype(np.int32)
        cv2.fillPoly(part_labels, pts=[pts], color=label_list[p])

    return keypoints, part_list, part_labels


def draw_face_edges(keypoints, part_list, part_labels, size, img):
    w, h = size
    edge_len = 3  
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8)  
    for edge_list in part_list:
        for edge in edge_list:
            for i in range(0, max(1, len(edge) - 1),
                           edge_len - 1):  
                sub_edge = edge[i:i + edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]

                curve_x, curve_y = interpPoints(x, y)  
                drawEdge(im_edges, curve_x, curve_y, bw=1, draw_end_points=False)

    canny_image = cv2.Canny(img, 80, 150)
    canny_image = canny_image * (part_labels == 0)
    im_edges += canny_image

    im_edges = image_crop(im_edges, keypoints, size)
    return im_edges


def func(x, a, b, c):
    return a * x**2 + b * x + c


def linear(x, a, b):
    return a * x + b


def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        else:
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]


def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)

def interpPoints(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)


def image_crop(img, keypoints, size):
    min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
    min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
    offset = (max_x - min_x) // 2
    min_y = max(0, min_y - offset * 2)
    min_x = max(0, min_x - offset)
    max_x = min(size[0], max_x + offset)
    max_y = min(size[1], max_y + offset)

    if isinstance(img, np.ndarray):
        return img[min_y: max_y, min_x:max_x]
    else:
        return img.crop((min_x, min_y, max_x, max_y))



def target_builder(img, size):
    points = face_keypoints_detector(img, MODEL_PATH)
    keypoints, part_list, part_labels = read_keypoints(points, size)
    im_edges = draw_face_edges(keypoints, part_list, part_labels, size, img)
    croped_image = image_crop(img, keypoints, size)
    return im_edges, croped_image



