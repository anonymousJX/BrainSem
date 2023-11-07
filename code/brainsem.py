import clip
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.vit import fMRI_ViT_Encoder
from fmri_handle.dataset_augment import *
from utils import *
from fmri_handle.GOD_dataset import *
from loss import *

def epoch_train(fmri_encoder, vit_clip, optimizer, train_dl):
    running_loss = []
    for idx, (images, fmris, categories) in enumerate(train_dl):
        fmri_encoder.train()
        images = images.to(DEVICE)
        fmris = fmris.float().to(DEVICE)
        category_labels = [(np.array(categories) == categories[i])[None,:] for i in range(len(categories))]
        category_labels = torch.tensor(np.vstack(category_labels)).to(DEVICE).float()
        soft_labels = category_labels / torch.sum(category_labels, 1)

        fmri_embedding = fmri_encoder(fmris)
        img_embedding = vit_clip.encode_image(images)
        fmri_embedding = fmri_embedding / fmri_embedding.norm(dim=-1, keepdim=True)
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
        loss, _ = ntxent_loss_with_soft_labels(fmri_embedding, img_embedding, soft_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

        if idx % 10 == 0: # print every 10 mini-batches
            trained_num = idx * images.shape[0]
            data_num = len(train_dl) * images.shape[0]
            percent = int(100. * trained_num / data_num)
            logging.info(f"Epoch: {(epoch + 1):4d} Batch: {(idx + 1):4d} [{(trained_num):5d}/{(data_num):5d} ({(percent):2d}%)]" +
                  f"  Loss: {(np.mean(running_loss)):.4f}")
            running_loss = []

if __name__ == '__main__':
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    selected_rois = ['ROI_V1', 'ROI_V2', 'ROI_V3', 'ROI_V4', 'ROI_LOC', 'ROI_FFA', 'ROI_PPA']
    selected_dataset = 'GOD'
    selected_sub = 'sub-3'
    mixup = True
    candidate = True
    log_path = setup_logging_from_args(f'./log/{selected_dataset}/{selected_sub}', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #ViT-B/16,ViT-B/32,RN50,RN101
    vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
    train_dl, test_dl, fmri_dim = get_image_fmri_dataset(selected_dataset, selected_sub, selected_rois, 200, mixup, candidate, preprocess)
    fmri_encoder = fMRI_ViT_Encoder(d = fmri_dim, num_patches = len(selected_rois), embed_dim = 512, depth = 24, num_heads = 16).to(DEVICE)
    #Large depth 24 heads 16 Base depth 12 heads 8 Small depth 6 heads 4
    optimizer = torch.optim.Adam(fmri_encoder.parameters(), lr = 1e-4)

    for epoch in range(5):
        epoch_train(fmri_encoder, vit_clip, optimizer, train_dl)