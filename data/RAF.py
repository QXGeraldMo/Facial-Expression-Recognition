import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torchvision
from torchvision import models
from functools import partial
import csv
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from PIL import Image

label_to_text = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}


class ToTensor(object):

    def __call__(self, sample):
        image, label, img_name, mode = sample['Image'], sample['Label'], sample['ImgName'], sample['Mode']

        image = np.expand_dims(image, axis=0)

        return {'Image': torch.from_numpy(image).float(), 'Label': label, 'ImgName': img_name, 'Mode': mode}


class RAF(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        df = pd.read_csv(csv_file, header=None, names=['column1'])
        df[['filename', 'label']] = df['column1'].str.split(' ', 1, expand=True)

        self.ImgNames = df.drop('column1', axis=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ImgNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.ImgNames.iloc[idx, 0])
        img_name = img_name.replace(".jpg", "")
        img_name = img_name + "_aligned" + ".jpg"
        label = self.ImgNames.iloc[idx, 1]
        label = int(label[0]) - 1

        #         image = Image.open(img_name).convert('RGB')
        image = Image.open(img_name).convert('L')
        image = np.array(image)
        #         image = io.imread(img_name)

        mode = ""
        if self.ImgNames.iloc[idx, 0].startswith("train"):
            mode = "train"
        if self.ImgNames.iloc[idx, 0].startswith("test"):
            mode = "test"

        sample = {'Image': image, 'Label': label, 'ImgName': img_name, "Mode": mode}

        if self.transform:
            sample = self.transform(sample)

        return sample


########################################################################################################
transformed_dataset = RAF(csv_file='./RAF/list_patition_label.txt',
                          root_dir='./RAF',
                          transform=transforms.Compose([ToTensor()]))

train_dataset = [sample for sample in transformed_dataset if sample['Mode'] == 'train']
test_dataset = [sample for sample in transformed_dataset if sample['Mode'] == 'test']

train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.25, random_state=42)

train_transform = transforms.Compose([
    transforms.RandomCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])

for sample in train_dataset:
    sample["Image"] = train_transform(sample["Image"])

test_transform = transforms.Compose([
    transforms.RandomCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])

for sample in val_dataset:
    sample["Image"] = test_transform(sample["Image"])

for sample in test_dataset:
    sample["Image"] = test_transform(sample["Image"])


