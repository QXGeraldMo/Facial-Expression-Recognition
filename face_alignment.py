import dlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from alignment.aligning import *
import torch

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("./alignment/shape_predictor_68_face_landmarks.dat")


def get_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    return img, height, width


def detect_face(img):
    faces = face_detector(img, 1)
    return faces


def detect_landmarks(img, faces, height, width):
    landmark_tuple = []
    cropped_imgs = []

    for k, d in enumerate(faces):
        landmarks = landmark_detector(img, d)
        left_eye = extract_left_eye_center(landmarks)
        right_eye = extract_right_eye_center(landmarks)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)
        cropped = crop_image(rotated, d)
        cropped_imgs.append(cropped)

        for n in range(0, 64):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))

    return cropped_imgs, landmark_tuple


def process_img(cropped_imgs, size=48):
    tensor_list = []
    # size = 100
    for cropped_img in cropped_imgs:
        img = Image.fromarray(cropped_img)
        img = img.resize((size, size))
        image_array = np.array(img)
        image_array = np.expand_dims(image_array, axis=0)
        tensor = torch.from_numpy(image_array).float()
        tensor_list.append(tensor)

    return tensor_list