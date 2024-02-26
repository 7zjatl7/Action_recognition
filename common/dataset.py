from torch.utils.data import Dataset, DataLoader, random_split
from .generator import *
import numpy as np
import torch

class ActionDataset(Dataset):
    def __init__(self, keypoints_dataset, transform=None):
        self.keypoints_dataset = keypoints_dataset
        self.keypoints = keypoints_dataset[0]
        self.labels = keypoints_dataset[1]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        keypoint = self.keypoints[idx]
        label = self.labels[idx]
        keypoint_tensor = torch.tensor(keypoint, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return keypoint_tensor, label_tensor

# class VideoActionDataset(Dataset):
#     # def __init__(self, landmarks, labels, transform=None):
#     def __init__(self, keypoint_dict):
#         self.samples = []
#         self.transform = transform

#         for video_name, data in keypoint_dict.items():
#             for keypoints in data['keypoints']:
#                 if transform:
#                     keypoints = transform(keypoints)
#                 self.samples.append((np.array(keypoints).flatten(), data['label']))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset 
