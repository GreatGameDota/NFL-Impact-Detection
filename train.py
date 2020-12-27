import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm,trange
import sklearn.metrics
import gc
import math
from glob import glob
from datetime import datetime
import time
import copy
import re

from sklearn.model_selection import train_test_split, KFold, GroupKFold, GroupShuffleSplit

os.system("pip install albumentations==0.5.2 -q")
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# from apex import amp

import warnings
warnings.filterwarnings("ignore")

if not os.path.isdir('data/images/'):
    os.system('python download.py')

from dataset import ImageDataset
from models import Context_FRCNN
from optimizers import get_optimizer
from schedulers import get_scheduler

from Config import config

from utilities import *
from evaluate import evaluate_model

import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def main():
    seed_everything(config.seed)
#     args = parse_args()

    ###### LOAD DATA ######

    train_df2 = pd.read_csv('data/train_labels_expanded.csv').fillna(0)
    train_df2 = train_df2.drop(train_df2[train_df2.frame==0].index)

    # video_labels_with_impact = initial_labels[initial_labels['impact']>0]
    # for row in tqdm(video_labels_with_impact[['video','frame','label']].values):
    #     frames = np.array([-4,-3,-2,-1,1,2,3,4])+row[1]
    #     initial_labels.loc[(initial_labels['video'] == row[0]) 
    #                                 & (initial_labels['frame'].isin(frames))
    #                                 & (initial_labels['label'] == row[2]), 'impact'] = 1
    # initial_labels.to_csv('train_labels_expanded.csv', index=False)

    initial_labels = pd.read_csv('data/train_labels.csv').fillna(0)
    
    vid = initial_labels[initial_labels.video=='58098_001193_Endzone.mp4']
    idxs = vid[vid.frame==40].index
    for idx in idxs:
        initial_labels.iloc[idx,10] = 0 # fix incorrect annotation

    vid = initial_labels[initial_labels.video=='57911_000147_Endzone.mp4']
    players = vid[vid.frame==113]
    player = players[players.label=='H21'].index
    initial_labels.iloc[player,10] = 0 # fix #2

    initial_labels.loc[(initial_labels.impact == 1) & (initial_labels.visibility == 0), 'impact'] = 0

    initial_labels = initial_labels.drop(initial_labels[initial_labels.frame==0].index)

    df = pd.DataFrame()
    df2 = pd.DataFrame()
    paths = os.listdir(config.IMAGE_PATH)
    paths.sort()
    for id in paths:
        df_ = train_df2[train_df2['video']==id+'.mp4']
        df_ = df_.drop_duplicates(subset='frame', keep='first').reset_index(drop=True)
        df = df.append(df_[config.max_frame:-config.max_frame]).reset_index(drop=True) # remove first and last frames

        df_ = initial_labels[initial_labels['video']==id+'.mp4']
        df_ = df_.drop_duplicates(subset='frame', keep='first').reset_index(drop=True)
        df2 = df2.append(df_[config.max_frame:-config.max_frame]).reset_index(drop=True)

    video_ids = []
    for id in df.video.values:
        id_ = id.split('_')
        video_ids.append(id_[0] + '_' + id_[1])
    video_ids = pd.DataFrame(video_ids, columns=['video_id'])
    df = pd.concat([df,video_ids], axis=1)

    video_ids = []
    for id in df2.video.values:
        id_ = id.split('_')
        video_ids.append(id_[0] + '_' + id_[1])
    video_ids = pd.DataFrame(video_ids, columns=['video_id'])
    df2 = pd.concat([df2,video_ids], axis=1)

    videos = df['video_id'].unique()
    indices = np.arange(len(videos))
    random.shuffle(indices)

    splits = list(GroupKFold(n_splits=config.folds).split(indices, groups=videos[indices]))

    folds = np.zeros(len(videos))
    df['fold'] = -1
    df2['fold'] = -1
    for fld, (_, test_idx) in enumerate(splits):
        folds[test_idx] = fld
    
    for i,p in enumerate(videos):
        df.loc[df.video==p+'_Endzone'+'.mp4','fold'] = folds[i]
        df.loc[df.video==p+'_Sideline'+'.mp4','fold'] = folds[i]

        df2.loc[df2.video==p+'_Endzone'+'.mp4','fold'] = folds[i]
        df2.loc[df2.video==p+'_Sideline'+'.mp4','fold'] = folds[i]
    
    #################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    additional_targets_dict = {}
    label_list = ['labels']
    for i in range(len(config.frame_idxs)):
        additional_targets_dict['image'+str(i+1)] = 'image'
        additional_targets_dict['bboxes'+str(i+1)] = 'bboxes'
        additional_targets_dict['labels'+str(i+1)] = f'labels{i+1}'
        label_list.append(f'labels{i+1}')
    
    train_transform = A.Compose([
                                #  A.RandomSizedCrop(min_max_height=(600, 600), height=config.image_size, width=config.image_size, p=0.5),
                                A.OneOf([
                                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                                        val_shift_limit=0.2, p=0.9),
                                    A.RandomBrightnessContrast(brightness_limit=0.2, 
                                                            contrast_limit=0.2, p=0.9),
                                ],p=0.9),
                                A.HorizontalFlip(p=0.5),
                                #  A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
                                A.OneOf([
                                    A.Blur(blur_limit=3, p=1.0),
                                    A.MedianBlur(blur_limit=3, p=1.0),
                                    A.MotionBlur(blur_limit=3, p=1.0)
                                ],p=0.1),
                                A.Resize(config.image_size,config.image_size,p=1),
                                #  A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
                                #  A.CenterCrop(512,512,p=1),
                                ToTensorV2(p=1)
    ],
    bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=label_list
    ),additional_targets=additional_targets_dict)

    val_transform = A.Compose([
                                A.Resize(config.image_size,config.image_size,p=1),
                                ToTensorV2(p=1)
    ],
    bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=label_list
    ),additional_targets=additional_targets_dict)

    folds = [0, 1, 2, 3, 4]
    
    log_name = f"../drive/My Drive/logs/log-{len(os.listdir('../drive/My Drive/logs/'))+1}.log"

    # Loop over folds
    for fld in range(1):
        # fold = config.single_fold
        fold = fld
        print('Train fold: %i'%(fold+1))
        with open(log_name, 'a') as f:
            f.write('Train Fold %i\n\n'%(fold+1))

        train_dataset = ImageDataset(df, train_df2, config.IMAGE_PATH, folds=[i for i in folds if i != fold], frames=config.frame_idxs, transform=train_transform)
        val_dataset = ImageDataset(df2, initial_labels, config.IMAGE_PATH, folds=[fold], frames=config.frame_idxs, transform=val_transform, mode='val')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            sampler=RandomSampler(train_dataset),
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            num_workers=4,
            shuffle=False,
            sampler=SequentialSampler(val_dataset),
            pin_memory=False,
            collate_fn=collate_fn,
        )
        
        # Build Model
        model = Context_FRCNN('resnet50', num_classes=config.classes, use_long_term_attention=True,
                      backbone_out_features=256, attention_features=256,
                      attention_post_rpn=True, attention_post_box_classifier=False, 
                      use_self_attention=False, self_attention_in_sequence=False, 
                      num_attention_heads=1, num_attention_layers=1)
        model = model.cuda()

        context_model = None
        # context_model = Context_FRCNN('resnet50', num_classes=config.classes, use_long_term_attention=True,
        #               return_context_feats=True,
        #               backbone_out_features=256, attention_features=256,
        #               attention_post_rpn=False, attention_post_box_classifier=False, 
        #               use_self_attention=False, self_attention_in_sequence=False, 
        #               num_attention_heads=1, num_attention_layers=1)
        # context_model.load_state_dict(torch.load('../drive/My Drive/Models/pretrained/frcnn-fld1 single.pth')['model_state'])
        # context_model.cuda()
        # context_model.eval()

        # Optimizer
        optimizer = get_optimizer(model, lr=config.lr)

        # Apex
        if config.apex:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O1', verbosity=0)

        # Training
        history = pd.DataFrame()
        history2 = pd.DataFrame()

        torch.cuda.empty_cache()
        gc.collect()

        best = 0
        best2 = 1e10
        n_epochs = config.epochs
        early_epoch = 0

        # Scheduler
        scheduler = get_scheduler(optimizer, train_loader=train_loader, epochs=n_epochs, batch_size=config.batch_size)
        updates_per_epoch = math.ceil(len(train_loader) / config.batch_size)

        for epoch in range(n_epochs-early_epoch):
            epoch += early_epoch
            torch.cuda.empty_cache()
            gc.collect()

            with open(log_name, 'a') as f:
                f.write(f'XXXXXXXXXXXXXX-- CYCLE INTER: {epoch+1} --XXXXXXXXXXXXXXXXXXX\n')
                lr_ = optimizer.state_dict()['param_groups'][0]['lr']
                f.write(f'curr lr: {lr_}\n')

            # ###################################################################
            # ############## TRAINING ###########################################
            # ###################################################################

            model.train()
            total_loss = 0

            t = tqdm(train_loader)
            for batch_idx, (img1, targets1, imgs2, targets2) in enumerate(t):
                img1 = torch.stack(img1)
                imgs2 = torch.stack(imgs2)
                img_batch1 = img1.cuda().float()
                img_batch2 = imgs2.cuda().float()
                targets1 = [{k: v.cuda() for k, v in t.items()} for t in targets1]
                targets3 = []
                for tar in targets2:
                    targets2_ = [{k: v.cuda() for k, v in t.items()} for t in tar]
                    targets3.append(targets2_)
                targets2 = targets3
                
                if context_model is not None:
                    context_feats, valid_size = context_model(context_images=img_batch2)

                rand = np.random.rand()
                if rand < config.mixup:
                    pass
                elif rand < config.cutmix:
                    pass
                else:
                    if config.scale:
                        with amp.autocast():
                            _, loss_dict = model(img_batch1, img_batch2, targets1, context_targets=targets2)
                            # _, loss_dict = model(img_batch1, context_features=context_feats, valid_context_size=valid_size, targets=targets1)
                    else:
                        _, loss_dict = model(img_batch1, img_batch2, targets1, context_targets=targets2)
                        # _, loss_dict = model(img_batch1, context_features=context_feats, valid_context_size=valid_size, targets=targets1)
                loss = sum(loss for loss in loss_dict.values())/ config.accumulation_steps

                total_loss += loss.data.cpu().numpy() * config.accumulation_steps
                t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(batch_idx+1)))

                if history is not None:
                    history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
                    history.loc[epoch + batch_idx / len(train_loader), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                
                if config.scale:
                    scaler.scale(loss).backward()
                elif config.apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if (batch_idx+1) % config.accumulation_steps == 0:
                    if config.scale:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # if scheduler is not None:
                #     scheduler.step(epoch)

            #### VALIDATION ####

            kaggle = evaluate_model(model, val_loader, epoch, scheduler=scheduler, history=history2, log_name=log_name)
            
            if kaggle > best:
                best = kaggle
                print(f'Saving best model... (metric)')
                torch.save({
                    'model_state': model.state_dict(),
                }, f'../drive/My Drive/Models/frcnn-fld{fold+1}.pth')
                with open(log_name, 'a') as f:
                    f.write('Saving Best model...\n\n')
            else:
                with open(log_name, 'a') as f:
                    f.write('\n')

if __name__ == '__main__':
    main()
