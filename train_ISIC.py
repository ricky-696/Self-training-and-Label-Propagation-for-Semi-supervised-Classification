# -*- coding: utf-8 -*-
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import random_split

from Model import resnet50, FC
from dataset import ISIC2018_Dataset

import trainer
from opt import arg_parse
from utils import get_logger


if __name__ == '__main__':
    args = arg_parse()
    # args.pretrain = False # debug
    args.dataset_dir = os.path.join('Datasets', 'ISIC2018')
    args.save_model_dir = os.path.join('trained_model', 'ISIC2018', 'resnet18')
    args.log_filename = 'train_ISIC2018'
    
    args.logger = get_logger(args.log_filename)
    args.device = torch.device(f'cuda:{args.devices[0]}' if torch.cuda.is_available() else 'cpu')
    args.logger.info(f'Start batch size: {args.batch_size}, device: {args.device}')

    os.makedirs('pic', exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
    ])

    args.num_classes = 7
    data_train = ISIC2018_Dataset(type='train', transform=transform)
    train_data, unlabeled_data = random_split(data_train, [0.5, 0.5])

    val_data = ISIC2018_Dataset(type='valid', transform=transform)

    args.train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=8
    )

    args.val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=8
    )
    
    args.unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=8
    )
    
    args.pretrain_model = resnet50(num_classes=args.num_classes).to(args.device)
    args.model_fc = FC(num_classes=args.num_classes).to(args.device) # used original's FC
    
    trainer.main(args)