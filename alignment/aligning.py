import numpy as np
import cv2

LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]


def extract_left_eye_center(shape):
    points = [shape.part(i) for i in LEFT_EYE_IDX]
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    center_x = sum(xs) // len(LEFT_EYE_IDX)
    center_y = sum(ys) // len(LEFT_EYE_IDX)
    return center_x, center_y


def extract_right_eye_center(shape):
    points = [shape.part(i) for i in RIGHT_EYE_IDX]
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    center_x = sum(xs) // len(RIGHT_EYE_IDX)
    center_y = sum(ys) // len(RIGHT_EYE_IDX)
    return center_x, center_y


def get_rotation_matrix(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def crop_image(image, det):
    left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()
    left = max(0, left)
    top = max(0, top)
    right = min(image.shape[1], right)
    bottom = min(image.shape[0], bottom)
    return image[top:bottom, left:right]
