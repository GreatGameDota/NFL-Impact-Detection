from utilities import *
from tqdm import tqdm,trange
import torch
from Config import config

def evaluate_model(model, val_loader, epoch, scheduler=None, history=None, log_name=None):
    model.eval()
    metric_ = 0.
    
    tps = []
    fps = []
    fns = []
    
    tps1 = []
    fps1 = []
    fns1 = []
    
    tps2 = []
    fps2 = []
    fns2 = []
    
    tps3 = []
    fps3 = []
    fns3 = []
    
    tps4 = []
    fps4 = []
    fns4 = []
    with torch.no_grad():
        t = tqdm(val_loader)
        for img1, targets1, imgs2, targets2, frames in t:
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

            dets, _ = model(img_batch1, img_batch2, targets1, targets2)
            
            for k,dets in enumerate(dets):
              boxes = []
              boxes1 = []
              boxes2 = []
              boxes3 = []
              boxes4 = []
              target_boxes = []
              for i in range(len(dets['boxes'])):
                det = dets['boxes'][i].cpu().numpy() # xyxy

                det[0] = (det[0] * config.input_W / config.image_size)
                det[1] = (det[1] * config.input_H / config.image_size)
                det[2] = (det[2] * config.input_W / config.image_size)
                det[3] = (det[3] * config.input_H / config.image_size)
                
                if dets['labels'][i].cpu().numpy().astype(int) == 2:
                  score = dets['scores'][i]
                  if score > 0.1:
                      boxes.append(det[0:4])
                      
                  if score > 0.2:
                      boxes1.append(det[0:4])
                  
                  if score > 0.3:
                      boxes2.append(det[0:4])

                  if score > 0.4:
                      boxes3.append(det[0:4])

                  if score > 0.5:
                      boxes4.append(det[0:4])

              for j,data in enumerate(targets1):
                for i in range(len(data['boxes'])):
                  if data['labels'][i].cpu().numpy() == 2:
                    box = data['boxes'][i].cpu().numpy() # yxyx
                    # box[[0,1,2,3]] = box[[1,0,3,2]] # xyxy

                    box[0] = (box[0] * config.input_W / config.image_size)
                    box[1] = (box[1] * config.input_H / config.image_size)
                    box[2] = (box[2] * config.input_W / config.image_size)
                    box[3] = (box[3] * config.input_H / config.image_size)
                    
                    target_boxes.append(box)
              
              clipped_target_boxes = []
              for idx,box in enumerate(target_boxes):
                new_box = [frames[k]]
                for val in box:
                  new_box.append(val)
                clipped_target_boxes.append(new_box)
              
              clipped_pred_boxes = []
              for idx,box in enumerate(boxes):
                new_box = [frames[k]]
                for val in box:
                  new_box.append(val)
                clipped_pred_boxes.append(new_box)
              
              tp, fp, fn = precision_calc(clipped_target_boxes, clipped_pred_boxes)
              tps.append(tp)
              fns.append(fn)
              fps.append(fp)

              clipped_pred_boxes = []
              for idx,box in enumerate(boxes1):
                new_box = [frames[k]]
                for val in box:
                  new_box.append(val)
                clipped_pred_boxes.append(new_box)
              
              tp, fp, fn = precision_calc(clipped_target_boxes, clipped_pred_boxes)
              tps1.append(tp)
              fns1.append(fn)
              fps1.append(fp)

              clipped_pred_boxes = []
              for idx,box in enumerate(boxes2):
                new_box = [frames[k]]
                for val in box:
                  new_box.append(val)
                clipped_pred_boxes.append(new_box)
              
              tp, fp, fn = precision_calc(clipped_target_boxes, clipped_pred_boxes)
              tps2.append(tp)
              fns2.append(fn)
              fps2.append(fp)

              clipped_pred_boxes = []
              for idx,box in enumerate(boxes3):
                new_box = [frames[k]]
                for val in box:
                  new_box.append(val)
                clipped_pred_boxes.append(new_box)
              
              tp, fp, fn = precision_calc(clipped_target_boxes, clipped_pred_boxes)
              tps3.append(tp)
              fns3.append(fn)
              fps3.append(fp)

              clipped_pred_boxes = []
              for idx,box in enumerate(boxes4):
                new_box = [frames[k]]
                for val in box:
                  new_box.append(val)
                clipped_pred_boxes.append(new_box)
              
              tp, fp, fn = precision_calc(clipped_target_boxes, clipped_pred_boxes)
              tps4.append(tp)
              fns4.append(fn)
              fps4.append(fp)
    
    tp = np.sum(tps)
    fp = np.sum(fps)
    fn = np.sum(fns)
    precision = tp / (tp + fp + 1e-6)
    recall =  tp / (tp + fn +1e-6)
    f1_score = 2*(precision*recall)/(precision+recall+1e-6)
    
    tp = np.sum(tps1)
    fp = np.sum(fps1)
    fn = np.sum(fns1)
    precision = tp / (tp + fp + 1e-6)
    recall =  tp / (tp + fn +1e-6)
    f1_score1 = 2*(precision*recall)/(precision+recall+1e-6)

    tp = np.sum(tps2)
    fp = np.sum(fps2)
    fn = np.sum(fns2)
    precision = tp / (tp + fp + 1e-6)
    recall =  tp / (tp + fn +1e-6)
    f1_score2 = 2*(precision*recall)/(precision+recall+1e-6)

    tp = np.sum(tps3)
    fp = np.sum(fps3)
    fn = np.sum(fns3)
    precision = tp / (tp + fp + 1e-6)
    recall =  tp / (tp + fn +1e-6)
    f1_score3 = 2*(precision*recall)/(precision+recall+1e-6)

    tp = np.sum(tps4)
    fp = np.sum(fps4)
    fn = np.sum(fns4)
    precision = tp / (tp + fp + 1e-6)
    recall =  tp / (tp + fn +1e-6)
    f1_score4 = 2*(precision*recall)/(precision+recall+1e-6)

    scores = [f1_score,f1_score1,f1_score2,f1_score3,f1_score4]

    if history is not None:
      # history.loc[epoch, 'val_loss'] = loss.cpu().numpy()
      history.loc[epoch, 'metric'] = max(scores)
    
    if scheduler is not None:
      scheduler.step(max(scores))

    print(f'F1 at .1: {f1_score}')
    print(f'F1 at .2: {f1_score1}')
    print(f'F1 at .3: {f1_score2}')
    print(f'F1 at .4: {f1_score3}')
    print(f'F1 at .5: {f1_score4}')
    
    with open(log_name, 'a') as f:
      f.write(f'val Metric: {max(scores)}\n')

    return max(scores)