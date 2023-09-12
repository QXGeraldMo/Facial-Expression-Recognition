import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils


# Process the dataset of FER2013+ with majority voting

class ToTensor(object):

    def __call__(self, sample):
        # image, label, img_name, label_dist = sample['Image'], sample['Label'], sample['ImgName'],sample['Label_Dist']
        image, label, img_name = sample['Image'], sample['Label'], sample['ImgName']

        #         image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        #         image = np.repeat(image, 3, axis=0)

        # return {'Image': torch.from_numpy(image).float(), 'Label':label, 'ImgName':img_name, 'Label_Dist':label_dist}
        return {'Image': torch.from_numpy(image).float(), 'Label': label, 'ImgName': img_name}


class Fer2013(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, mode="train"):
        self.ImgNames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.ImgNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.ImgNames.iloc[idx, 0])
        image = io.imread(img_name)
        #         image = Image.open(img_name).convert('LA').convert('RGB')
        #         image= np.array(image)

        label_dist = np.array(self.ImgNames.iloc[idx, 2:-2].values.tolist())
        label_dist = np.divide(label_dist, 10)

        label = np.argmax(label_dist)

        #         max_index = np.argmax(label_dist)
        #         label = np.zeros_like(label_dist)
        #         label[max_index] = 1

        train_transform = transforms.Compose([
            transforms.RandomCrop(44),
            transforms.RandomRotation(degrees=(-25, 25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()])

        val_transform = transforms.Compose([
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()])

        test_transform = transforms.Compose([
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()])

        #         sample = {'Image':image, 'Label':label, 'ImgName':img_name, 'Label_Dist':label_dist}
        sample = {'Image': image, 'Label': label, 'ImgName': img_name}

        if self.transform:
            sample = self.transform(sample)

            if self.mode == "train":
                sample["Image"] = train_transform(sample["Image"])
            if self.mode == "val":
                sample["Image"] = val_transform(sample["Image"])
            if self.mode == "test":
                sample["Image"] = test_transform(sample["Image"])

        return sample

########################################################################
########################################################################

# train_dataset = Fer2013(csv_file='./FER2013plus/FER2013Train/label.csv',
#                         root_dir='./FER2013plus/FER2013Train',
#                         transform=transforms.Compose([ToTensor()]),
#                         mode="train")
#
# validation_dataset = Fer2013(csv_file='./FER2013plus/FER2013Valid/label.csv',
#                              root_dir='./FER2013plus/FER2013Valid',
#                              transform=transforms.Compose([ToTensor()]),
#                              mode = 'val')
#
# test_dataset = Fer2013(csv_file='./FER2013plus/FER2013Test/label.csv',
#                        root_dir='./FER2013plus/FER2013Test',
#                        transform=transforms.Compose([ToTensor()]),
#                        mode="test")
