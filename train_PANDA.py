# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import random_split

from Model import *

import trainer
from opt import arg_parse
from utils import get_logger

from Kaggle_PANDA_1st_place_solution.src.factory import get_transform, read_yaml
from Kaggle_PANDA_1st_place_solution.src.dataset import TrainDataset
from Kaggle_PANDA_1st_place_solution.src.models.efficientnet import CustomEfficientNet

if __name__ == '__main__':
    args = arg_parse()
    PANDA_cfg = read_yaml(fpath='Kaggle_PANDA_1st_place_solution/src/configs/final_1.yaml')
    transform = get_transform(conf_augmentation=PANDA_cfg.Augmentation['train'])

    train_set = TrainDataset(
        conf_dataset=PANDA_cfg.Data.dataset,
        phase='train',
        out_ch=PANDA_cfg.Model.out_channel,
        transform=transform,
    )

