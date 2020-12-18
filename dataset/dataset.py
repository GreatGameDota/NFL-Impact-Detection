import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pandas as pd

class ImageDataset(Dataset):
  def __init__(self, df, train_df, root_dir, folds, frames, transform=None, mode='train'):
    self.df = df[df.fold.isin(folds).reset_index(drop=True)].reset_index(drop=True)
    self.train_df = train_df
    self.root_dir = root_dir
    self.folds = folds
    self.transform = transform
    self.mode = mode
    self.frames = frames

    self.videos = self.df.video.unique()
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    # print(idx)
    data = self.df.iloc[idx]
    video_name = data['video']
    frame = data['frame']
    img = cv2.imread(f'{self.root_dir}{video_name[:-4]}/{frame-1}.jpg')
    if img is None:
      print(f'{self.root_dir}{video_name[:-4]}/{frame}.jpg')
    
    data2 = self.train_df[self.train_df.video==video_name]
    data3 = data2[data2.frame == frame]
    boxes = data3[['left','width','top','height']].values
    boxes2 = boxes.copy()
    boxes2[:,1] = boxes[:,2] + boxes[:,3]     # bottom = top + height
    boxes2[:,3] = boxes[:,2]                  # move top to last
    boxes2[:,2] = boxes[:,0] + boxes[:,1]     # right = left + width (xyxy)
    boxes2[:,[0,1,2,3]] = boxes2[:,[0,3,2,1]] # swap y's

    labels = data3['impact'].values

    # idxs = np.array([-2,2])
    frames = frame + self.frames
    # frames = [-1,frames[0]]
    total_frames = len(self.df[self.df.video==video_name].frame.values) + self.frames[-1]*2 # first and last frames
    imgs2 = []
    labels2 = []
    boxes4 = []
    for fr in frames:
      if fr > 0 and fr < total_frames+1:
        img2 = cv2.imread(f'{self.root_dir}{video_name[:-4]}/{fr-1}.jpg')
        if img2 is None:
          print(f'{self.root_dir}{video_name[:-4]}/{fr}.jpg')
        imgs2.append(img2)

        data3 = data2[data2.frame == fr]
        boxes = data3[['left','width','top','height']].values
        boxes3 = boxes.copy()
        boxes3[:,1] = boxes[:,2] + boxes[:,3]     # bottom = top + height
        boxes3[:,3] = boxes[:,2]                  # move top to last
        boxes3[:,2] = boxes[:,0] + boxes[:,1]     # right = left + width (xyxy)
        boxes3[:,[0,1,2,3]] = boxes3[:,[0,3,2,1]] # swap y's
        boxes4.append(boxes3)
        # labels2.append(np.zeros((len(data3['impact'].values),)))
        labels2.append(data3['impact'].values)
      else:
        print('err')
        boxes4.append([0,0,.1,.1])
        labels2.append([0])

    max_imgs2 = []
    for lab in labels2:
      max_imgs2.append(len(lab))
    
    boxes_len = max([len(labels),max(max_imgs2)])
    for i in range(boxes_len - len(labels)):
      boxes2 = np.vstack([boxes2,[0,0,0.1,0.1]])
      labels = np.append(labels,[0])

    for j,boxes_ in enumerate(boxes4):
      for i in range(boxes_len - len(boxes_)):
        boxes4[j] = np.vstack([boxes4[j],[0,0,0.1,0.1]])
        labels2[j] = np.append(labels2[j],[0])
    
    # print(boxes2.shape, len(labels))
    # print(len(boxes4[0]), len(labels2[0]))
    # print(len(boxes4[1]), len(labels2[1]))

    name = 'image'
    aug_imgs = {}
    for i,im in enumerate(imgs2):
      aug_imgs[name+str(i+1)] = im
    name = 'bboxes'
    aug_boxes = {}
    for i,box in enumerate(boxes4):
      aug_boxes[name+str(i+1)] = box
    name = 'labels'
    aug_labels = {}
    for i,l in enumerate(labels2):
      aug_labels[name+str(i+1)] = l
    
    if self.transform is not None:
      for i in range(10): # loop until crop with bounding boxes is found
        aug = self.transform(**{
            'image': img,
            'bboxes': boxes2,
            'labels': labels,
        },**aug_imgs, **aug_boxes, **aug_labels)
        if len(aug['bboxes']) > 0:
          img = aug['image'] / 255.
          boxes2 = torch.tensor(aug['bboxes'])
          # boxes2[:,[0,1,2,3]] = boxes2[:,[1,0,3,2]] # swap x's and y's (yxyx)

          # boxes3_ = []
          # remove_idxs = []
          # for i,box in enumerate(boxes2):
          #   if (box[3]-box[1])*(box[2]-box[0]) > .1:
          #     boxes3_.append(box)
          #   else:
          #     remove_idxs.append(i)
          # boxes2 = torch.stack(boxes3_)

          labels = aug['labels']
          labels_ = torch.ones((len(boxes2),), dtype=torch.int64)
          for i,label in enumerate(labels):
            if label == 1:
              labels_[i] = 2
          # for i in remove_idxs:
          #   labels_ = torch.cat([labels_[0:i], labels_[i+1:]])
          
          imgs2 = []
          for name in aug_imgs.keys():
            imgs2.append(aug[name] / 255.)
          imgs2 = torch.stack(imgs2)

          boxes4 = []
          remove_idxs = []
          for j,name in enumerate(aug_boxes.keys()):
            boxes2_ = torch.tensor(aug[name])
            # boxes2_[:,[0,1,2,3]] = boxes2_[:,[1,0,3,2]] # swap x's and y's (yxyx)

            # boxes3_ = []
            # for i,box in enumerate(boxes2_):
            #   if (box[3]-box[1])*(box[2]-box[0]) > .1:
            #     boxes3_.append(box)
            #   else:
            #     remove_idxs.append([j,i])
            # boxes2_ = torch.stack(boxes3_)
            boxes4.append(boxes2_)
          
          labels2 = []
          for j,name in enumerate(aug_labels.keys()):
            labels = aug[name]
            labels2_ = torch.ones((len(boxes2),), dtype=torch.int64)
            for i,label in enumerate(labels):
              if label == 1:
                labels2_[i] = 2
            # for i in remove_idxs:
            #   if i[0] == j:
            #     labels2_ = torch.cat([labels2_[0:i[1]], labels2_[i[1]+1:]])
            labels2.append(labels2_)
          
          targets2 = []
          for i in range(len(boxes4)):
            target = {}
            target['boxes'] = boxes4[i]
            target['labels'] = labels2[i]
            targets2.append(target)
          
          break
    if self.mode == 'val':
      return img, {'boxes':boxes2, 'labels':labels_}, imgs2, targets2, frame
    return img, {'boxes':boxes2, 'labels':labels_}, imgs2, targets2
