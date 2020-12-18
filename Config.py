import numpy as np

class config:
    epochs = 2
    batch_size = 4
    input_H = 720
    input_W = 1280
    image_size = 512
    classes = 3
    frame_idxs = np.array([-1,1])
    max_frame = frame_idxs[-1]
    IMAGE_PATH = 'data/images/'
    TEST_PATH = 'test/'
    # lr = 1e-4
    lr = 0.0002
    # lr = 0.003
    # lr = 3e-4
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    seed = 42
    mixup = 0
    cutmix = 0
    accumulation_steps = 1
    single_fold = 0
    folds = 5
    apex = False
    scale = False # doesnt work in kaggle kernals
