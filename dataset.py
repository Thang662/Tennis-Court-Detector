from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import cv2
from utils import line_intersection
from heatmap import gen_binary_map
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_pil_image as ToPILImage
from torch.utils.data import DataLoader
import os
import json
import time

class Tennis(Dataset):
    """
    """
    def __init__(self, root, train, frame_in, is_sequential, transform = None,  r = 2.5, w = 512, h = 288):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.frame_in = frame_in
        self.is_sequential = is_sequential
        self.data = self.load_data()
        self.r = r
        self.w = w
        self.h = h
        if self.transform is None:
            if self.train:
                self.transform = A.Compose([
                        A.Resize(height = self.h, width = self.w, p = 1),
                        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255.0, p = 1.0),
                        ToTensorV2()
                    ],
                    keypoint_params = A.KeypointParams(format = 'xy'),
                )
            else:
                self.transform = A.Compose([
                        A.Resize(height = self.h, width = self.w, p = 1),
                        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255.0, p = 1.0),
                        ToTensorV2()
                    ],
                    keypoint_params = A.KeypointParams(format = 'xy'),
                )

    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        with open(self.root / f'data_{"train" if self.train else "val"}.json', 'r') as f:
            data = json.load(f)
        tmp_data = []
        step = 1 if self.is_sequential else self.frame_in
        for i in range(0, len(data) - self.frame_in + 1, step):
            paths = [str(self.root / 'images' / f'{d["id"]}.png') for d in data[i:i + self.frame_in]]
            keypoints = [d['kps'] for d in data[i:i + self.frame_in]]
            tmp_data.append((paths, keypoints))
        return tmp_data
    
    def __getitem__(self, index):
        paths, keypoints = self.data[index]
        imgs = []
        heat_maps = []
        annos = []
        annos_transformed = []
        vises = []
        for path, keypoint in zip(paths, keypoints):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image = img, keypoints = keypoint)
            img = transformed['image']
            keypoint_transformed = transformed['keypoints']
            x_ct, y_ct = line_intersection((keypoint_transformed[0][0], keypoint_transformed[0][1], keypoint_transformed[3][0], keypoint_transformed[3][1]), 
                                           (keypoint_transformed[1][0], keypoint_transformed[1][1], keypoint_transformed[2][0], keypoint_transformed[2][1]))
            keypoint_transformed = keypoint_transformed + [[float(x_ct), float(y_ct)]]
            tmp_heat_maps = [gen_binary_map((img.shape[2], img.shape[1]), kp, self.r) for kp in keypoint_transformed]
            imgs.append(img)
            heat_maps.append(torch.tensor(np.array(tmp_heat_maps)))
            annos_transformed.append(torch.tensor(keypoint_transformed))
        imgs = torch.cat(imgs)
        heat_maps = torch.cat(heat_maps)
        annos_transformed = torch.cat(annos_transformed)
        return imgs.float(), heat_maps.float(), annos_transformed.float()
    
def get_data_loaders(root, frame_in, is_sequential, batch_size, transform = None, NUM_WORKERS = os.cpu_count()):
    train_dataset = Tennis(root = root, train = True, transform = transform, frame_in = frame_in, is_sequential = is_sequential)
    test_dataset = Tennis(root = root, train = False, transform = transform, frame_in = frame_in, is_sequential = is_sequential)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = False)
    return train_loader, test_loader
    


if __name__ == "__main__":
    root = "D:\\thang\\20232\\thesis\\Dataset\\Dataset"
    train = True
    # transform = A.Compose([
    #     A.Resize(288, 512, p = 1),
    #     A.RandomBrightnessContrast(p = 0.2),
    #     A.HorizontalFlip(p = 0.5),
    #     A.VerticalFlip(p = 0.5),
    #     A.Rotate(limit = 40, p = 0.9),
    #     A.RandomSizedCrop(height = int(288 * 0.8), width = int(512 * 0.8), p = 0.9),
    #     A.Resize(288, 512, p = 1),
    #     A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255.0, p = 1.0),
    #     ToTensorV2()
    # ], keypoint_params = A.KeypointParams(format = 'xy', remove_invisible = True, angle_in_degrees = True))
    transform = None
    frame_in = 3
    is_sequential = False
    dataset = Tennis(root, train = train, transform = transform, frame_in = frame_in, is_sequential = is_sequential)
    # dataset[200]
    train_loader, test_loader = get_data_loaders(root = root, transform = transform, frame_in = frame_in, is_sequential = is_sequential, batch_size = 2, NUM_WORKERS = 2)
    # for i, (imgs, heat_maps, annos, annos_transformed) in enumerate(test_loader):
    #     # print(imgs, heat_maps, annos, annos_transformed)
    #     print(imgs.shape, heat_maps.shape, annos.shape, annos_transformed.shape)
    print(next(iter(train_loader))[4])