import dlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from face_alignment import *
from predicting import *


def camera(model):
    capture = cv2.VideoCapture(0)
    label_to_text = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear',
                     7: 'contempt'}

    while True:
        _, frame = capture.read()
        frame = cv2.cvtColor(cv2.resize(frame, (800, 600)), cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector(frame_gray)
        tensor_list = []
        cropped_img_list = []

        if faces is not None and len(faces) > 0:
            for k, d in enumerate(faces):
                x, y, w, h = d.left(), d.top(), d.width(), d.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                landmarks = landmark_detector(frame_gray, d)
                left_eye = extract_left_eye_center(landmarks)
                right_eye = extract_right_eye_center(landmarks)
                height, width = frame_gray.shape[:2]
                M = get_rotation_matrix(left_eye, right_eye)
                rotated = cv2.warpAffine(frame_gray, M, (width, height), flags=cv2.INTER_CUBIC)
                cropped = crop_image(rotated, d)
                cropped_img_list.append(cropped)
                tensor_list = process_img(cropped_img_list)

        predicted_labels, _ = predicting(tensor_list, model)

        text = ""
        for label in predicted_labels:
            text = text + " " + label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 255, 0)  # BGR color format
        thickness = 2
        position = (50, 50)
        # Add the text to the image
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

        cv2.imshow("FER", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(30)

        if key == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
