# -*- coding: utf-8 -*-
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split

from Model import gray_resnet18, Network
from dataset import MNIST_omega

import trainer
from opt import arg_parse
from utils import get_logger


if __name__ == '__main__':
    args = arg_parse()
    
    args.logger = get_logger(args.log_filename)
    args.device = torch.device(f'cuda:{args.devices[0]}' if torch.cuda.is_available() else 'cpu')
    args.logger.info(f'Start batch size: {args.batch_size}, device: {args.device}')

    os.makedirs('pic', exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
    ])

    data_train = MNIST_omega(
        root=args.dataset_dir,
        train=True,
        transform=transform,
        download=True,
        debug=False
    )  
    
    data_test = MNIST_omega(
        root=args.dataset_dir,
        train=False,
        transform=transform,
        debug=False
    )

    args.num_classes = data_train.get_num_classes()

    train_data, unlabeled_data = random_split(data_train, [0.1, 0.9])
    val_data = data_test

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
    
    args.pretrain_model = gray_resnet18(num_classes=args.num_classes).to(args.device)
    args.model_fc = Network().to(args.device) # used original's FC
    
    trainer.main(args)