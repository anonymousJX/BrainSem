import os
import torch
import bdpy
import numpy as np
import pandas as pd
from PIL import Image
import fmri_handle.configure as config
from torch.utils.data import Dataset, DataLoader

def GOD_sub_with_images(img_transfomers, sub = 'sub-3', rois = ['ROI_VC']):
    train_rois, test_rois, test_avg_rois = [], [], []
    GenericObjectDecoding_sub = bdpy.BData(os.path.join(config.GenericObjectDecoding_dataset_dir, 
                                                        config.GenericObjectDecoding_subs[sub]))
    DataType = GenericObjectDecoding_sub.select('DataType')
    train_index = (DataType == 1).squeeze()
    test_index = (DataType == 2).squeeze()

    image_index = GenericObjectDecoding_sub.select('image_index')
    train_image_index = image_index[train_index, :].squeeze().astype(int) - 1
    test_image_index = image_index[test_index, :].squeeze().astype(int) - 1
    trainStiIDs = np.array(pd.read_csv(config.kamitani_sti_trainID, header = None)[1])[train_image_index]
    testStiIDs = np.array(pd.read_csv(config.kamitani_sti_testID, header = None)[1])
    trainStis = torch.cat([img_transfomers(Image.open(f"{config.kamitani_Sti}/train/{file}"))[None,:,:,:] for file in trainStiIDs])
    testStis = torch.cat([img_transfomers(Image.open(f"{config.kamitani_Sti}/test/{file}"))[None,:,:,:] for file in testStiIDs])

    MAX_DIM = 0
    for roi in rois:
        roi_fMRI = GenericObjectDecoding_sub.select(roi)
        train_roi_fMRI = roi_fMRI[train_index, :]
        test_roi_fMRI = roi_fMRI[test_index, :]

        test_roi_fMRI_avg = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_roi_fMRI_avg[i] = np.mean(test_roi_fMRI[test_image_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI)
        test_avg_rois.append(test_roi_fMRI_avg)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in train_rois]), 1).squeeze()
    test_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_rois]), 1).squeeze()
    test_avg_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_avg_rois]), 1).squeeze()

    return train_rois, trainStis, test_avg_rois, testStis#, test_rois, test_image_index

def GOD_subs_with_images(test_sub = ['sub-3'], image_size = 128, roi = 'ROI_VC', batch_size = 10):
    fMRI_train, img_train, fMRI_test, img_test = [], [], [], []
    for sub in config.GenericObjectDecoding_subs:
        train_roi_fMRI, train_labels, test_roi_fMRI_avg, test_avg_labels = \
                GOD_sub_with_images(image_size = image_size, sub = sub, roi = roi)
        fMRI_train.append(train_roi_fMRI)
        img_train.append(train_labels)
        if(sub in test_sub):
            fMRI_test.append(test_roi_fMRI_avg)
            img_test.append(test_avg_labels)
        else:
            fMRI_train.append(test_roi_fMRI_avg)
            img_train.append(test_avg_labels)

    MAX_DIM = config.GOD_fMRI_dim['sub-3']
    img_train = np.vstack(img_train).astype(np.float32)
    img_test = np.vstack(img_test).astype(np.float32)
    fMRI_train = np.vstack(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1]))) for fmri in fMRI_train])).astype(np.float32)
    fMRI_test = np.vstack((([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1]))) for fmri in fMRI_test]))).astype(np.float32)

    train_dataset = config.fMRI_natural_Dataset(fMRI_train, img_train, range(img_train.shape[0]))
    test_dataset = config.fMRI_natural_Dataset(fMRI_test, img_test, range(img_test.shape[0]))
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader

def GOD_sub_without_images(sub = 'sub-3', rois = ['ROI_VC']):
    train_rois, test_rois = [], []
    GenericObjectDecoding_sub = bdpy.BData(os.path.join(config.GenericObjectDecoding_dataset_dir, 
                                                        config.GenericObjectDecoding_subs[sub]))
    DataType = GenericObjectDecoding_sub.select('DataType')
    train_index = (DataType == 1).squeeze()
    test_index = (DataType == 2).squeeze()

    image_index = GenericObjectDecoding_sub.select('image_index')
    train_image_index = image_index[train_index, :].squeeze().astype(int) - 1
    test_image_index = image_index[test_index, :].squeeze().astype(int) - 1
    trainStiIDs = np.array(pd.read_csv(config.kamitani_sti_trainID, header = None)[1])[train_image_index]
    testStiIDs = np.array(pd.read_csv(config.kamitani_sti_testID, header = None)[1])
    
    MAX_DIM = 0
    for roi in rois:
        roi_fMRI = GenericObjectDecoding_sub.select(roi)
        train_roi_fMRI = roi_fMRI[train_index, :]
        test_roi_fMRI = roi_fMRI[test_index, :]

        test_roi_fMRI_avg = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_roi_fMRI_avg[i] = np.mean(test_roi_fMRI[test_image_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI_avg)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in train_rois]), 1).squeeze()
    test_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_rois]), 1).squeeze()

    train_cat_rois = {}
    trainCatIDs = [id.split('_')[0] for id in trainStiIDs]
    trainCatSet = set(trainCatIDs)
    for cat in trainCatSet:
        train_cat_rois[cat] = train_rois[np.array(trainCatIDs) == cat]

    test_cat_rois = {}
    testCatIDs = [id.split('_')[0] for id in testStiIDs]
    for cat in testCatIDs:
        test_cat_rois[cat] = test_rois[np.array(testCatIDs) == cat]

    return train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs


def GOD_subs_without_images(subs = ['sub-3'], rois = ['ROI_VC']):
    train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs = [], [], [], [], [], []
    
    MAX_DIM = 0
    for sub in subs:
        _, sub_train_rois, sub_trainStiIDs, \
        _, sub_test_rois, sub_testStiIDs = GOD_sub_without_images(sub = sub, rois = rois)
        train_rois.append(sub_train_rois)
        trainStiIDs.append(sub_trainStiIDs)
        test_rois.append(sub_test_rois)
        testStiIDs.append(sub_testStiIDs)
        MAX_DIM = sub_train_rois.shape[-1] if sub_train_rois.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.vstack(([np.pad(fmri, ((0, 0), (0, 0), (0, MAX_DIM - fmri.shape[-1]))) for fmri in train_rois]))
    test_rois = np.vstack(([np.pad(fmri, ((0, 0), (0, 0), (0, MAX_DIM - fmri.shape[-1]))) for fmri in test_rois]))
    trainStiIDs = np.hstack(trainStiIDs)
    testStiIDs = np.hstack(testStiIDs)
    
    train_cat_rois = {}
    trainCatIDs = [id.split('_')[0] for id in trainStiIDs]
    trainCatSet = set(trainCatIDs)
    for cat in trainCatSet:
        train_cat_rois[cat] = train_rois[np.array(trainCatIDs) == cat]

    test_cat_rois = {}
    testCatIDs = [id.split('_')[0] for id in testStiIDs]
    testCatSet = set(testCatIDs)
    for cat in testCatSet:
        test_cat_rois[cat] = test_rois[np.array(testCatIDs) == cat]

    return train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs

def GOD_across_subs_without_images(sub_id = 3, rois = ['ROI_VC']):
    train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs, MAX_DIM = [], [], [], [], [], [], 0
    for sub in ['sub-1', 'sub-2', 'sub-3', 'sub-4', 'sub-5']:
        _, sub_train_rois, sub_trainStiIDs, \
        _, sub_test_rois, sub_testStiIDs = GOD_sub_without_images(sub = sub, rois = rois)
        train_rois.append(sub_train_rois)
        trainStiIDs.append(sub_trainStiIDs)
        test_rois.append(sub_test_rois)
        testStiIDs.append(sub_testStiIDs)
        MAX_DIM = sub_train_rois.shape[-1] if sub_train_rois.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = [np.pad(fmri, ((0, 0), (0, 0), (0, MAX_DIM - fmri.shape[-1]))) for fmri in train_rois]
    train_rois.pop(sub_id - 1)
    train_rois = np.vstack(train_rois)
    trainStiIDs.pop(sub_id - 1)
    trainStiIDs = np.hstack(trainStiIDs)
    test_rois = [np.pad(fmri, ((0, 0), (0, 0), (0, MAX_DIM - fmri.shape[-1]))) for fmri in test_rois]
    test_rois = test_rois[sub_id - 1]
    testStiIDs = testStiIDs[sub_id - 1]
    
    train_cat_rois = {}
    trainCatIDs = [id.split('_')[0] for id in trainStiIDs]
    trainCatSet = set(trainCatIDs)
    for cat in trainCatSet:
        train_cat_rois[cat] = train_rois[np.array(trainCatIDs) == cat]

    test_cat_rois = {}
    testCatIDs = [id.split('_')[0] for id in testStiIDs]
    testCatSet = set(testCatIDs)
    for cat in testCatSet:
        test_cat_rois[cat] = test_rois[np.array(testCatIDs) == cat]

    return train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs
