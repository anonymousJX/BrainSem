import os
import bdpy
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data
import fmri_handle.configure as config
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io.image as imageio
import fmri_handle.GOD_dataset as GOD

class Image_fMRI_Dataset(Dataset):
     def __init__(self, imageIDs, fMRI, mixup = False, train = True, transform = None):
         self.imageIDs = imageIDs
         self.fMRI = fMRI
         self.mixup = mixup
         self.train = train
         self.transform = transform
         
     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        category_id = file_name.split('_')[0]
        image_id = file_name.split('_')[1].split('.')[0]
        image = Image.open(self.imageIDs[idx])

        if(self.train):
            fmri_num = self.fMRI[category_id].shape[0]
            selected_no = np.random.permutation(range(fmri_num))[:random.randint(1, fmri_num - 1)]
            if(self.mixup and selected_no.shape[0] != 1):
                coefficient = torch.tensor(np.random.uniform(-1, 1, size = selected_no.shape[0])).softmax(0)
                mixup_fMRI = torch.sum(torch.tensor(self.fMRI[category_id][selected_no]) * coefficient[:, None, None], 0)
                fMRI = mixup_fMRI.numpy()
            else:
                fMRI = self.fMRI[category_id][selected_no[0]].squeeze()
        else:
            selected = random.randint(0, self.fMRI[category_id].shape[0] - 1)
            fMRI = self.fMRI[category_id][selected]

        image = self.transform(image) if self.transform else image
        return image, fMRI, category_id

def get_image_fmri_dataset(dataset, sub, rois, batch_size, mixup = True, candidate = True, img_transform = None):
    if(dataset == 'GOD'):
        train_cat_rois, _, trainStiIDs, test_cat_rois, _, testStiIDs = GOD.GOD_sub_without_images(sub, rois)
    else:
        raise NotImplementedError
    fmri_dim = train_cat_rois[trainStiIDs[0].split('_')[0]].shape[-1]
    Train_category = set([id.split('_')[0] for id in trainStiIDs])
    Test_category = set([id.split('_')[0] for id in testStiIDs])
    if(candidate):
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Train_category])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Test_category])
    else:
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in trainStiIDs])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in testStiIDs])
        
    train_dataset = Image_fMRI_Dataset(train_images, train_cat_rois, mixup, True, img_transform)
    test_dataset = Image_fMRI_Dataset(test_images, test_cat_rois, mixup, False, img_transform)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
    return train_dataloader, test_dataloader, fmri_dim