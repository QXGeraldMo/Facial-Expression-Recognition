import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def predicting(tensor_list, model):
    label_to_text = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear',
                     7: 'contempt'}
    # label_to_text = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}

    predicted_labels = []
    scores = []

    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    for tensor in tensor_list:
        tensor = tensor.to(device)
        model = model.to(device)
        c, h, w = np.shape(tensor)
        tensor = tensor.view(-1, c, h, w)  ##torch.Size([1, 1, 48, 48])
        output = model(tensor)

        score = F.softmax(output)
        scores.append(score)

        _, predicted1 = torch.max(output.data, 1)
        predicted_value = predicted1.item()
        predicted_label = label_to_text[predicted_value]
        predicted_labels.append(predicted_label)

    return predicted_labels, scores
