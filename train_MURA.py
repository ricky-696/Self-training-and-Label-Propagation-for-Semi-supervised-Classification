# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import random_split

from Model import *
from dataset import MURAv1_1
# from DenseNet_MURA_PyTorch.densenet import densenet169

import trainer
from opt import arg_parse
from utils import get_logger


def train_parts_separately(label_ratio, title):
    for study_type in args.study_type:
        args.dataset_dir = os.path.join('Datasets', 'MURA-v1.1')
        args.save_model_dir = os.path.join('trained_model', title, study_type, args.model_type)
        args.log_filename = title + study_type
        
        args.logger = get_logger(args.log_filename)
        args.device = torch.device(f'cuda:{args.devices[0]}' if torch.cuda.is_available() else 'cpu')

        args.logger.info(f'Start Traing: {study_type} dataset')
        args.logger.info(f'Start batch size: {args.batch_size}, device: {args.device}')

        os.makedirs('pic', exist_ok=True)
        
        args.num_classes = 2
        data_train = MURAv1_1(type='train', study_type=study_type, transform=args.data_transforms['train'])
        
        train_data, unlabeled_data = random_split(data_train, [label_ratio, 1 - label_ratio])

        val_data = MURAv1_1(type='valid', study_type=study_type, transform=args.data_transforms['valid'])

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
        
        if args.model_type == 'resnet':
            args.pretrain_model = resnet50(args.num_classes).to(args.device)
        elif args.model_type == 'densenet':
            args.pretrain_model = DenseNet169_BC(pretrain=False).to(args.device)
            
        args.model_fc = Same_Label(num_classes=args.num_classes, device=args.device).to(args.device)
        args.criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        args.optimizer = torch.optim.Adam(args.pretrain_model.parameters(), lr=0.001)
        
        trainer.main(args)


def train_all_parts(args, num_label=1000, title='train_MURA-v1.1_1000_all_parts_epoch_10_'):
    args.dataset_dir = os.path.join('Datasets', 'MURA-v1.1')
    args.save_model_dir = os.path.join('trained_model', title, 'all_parts', args.model_type)
    args.log_filename = title + 'all_parts'
    
    args.logger = get_logger(args.log_filename)
    args.device = torch.device(f'cuda:{args.devices[0]}' if torch.cuda.is_available() else 'cpu')

    args.logger.info(f'Start Traing: all_parts dataset')
    args.logger.info(f'Start batch size: {args.batch_size}, device: {args.device}')

    os.makedirs('pic', exist_ok=True)
    
    args.num_classes = 2
    data_train = MURAv1_1(type='train', all_parts=True, transform=args.data_transforms['train'])
    train_data, unlabeled_data = random_split(data_train, [num_label, len(data_train) - num_label])
    val_data = MURAv1_1(type='valid', all_parts=True, transform=args.data_transforms['valid'])

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
    
    if args.model_type == 'resnet':
        args.pretrain_model = resnet50(args.num_classes).to(args.device)
    elif args.model_type == 'densenet':
        args.pretrain_model = DenseNet169_BC(pretrain=False).to(args.device)
        
    args.model_fc = Same_Label(num_classes=args.num_classes, device=args.device).to(args.device)
    args.criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    args.optimizer = torch.optim.Adam(args.pretrain_model.parameters(), lr=0.001)

    # https://arxiv.org/pdf/2103.02193.pdf setting
    # args.optimizer = torch.optim.SGD(args.pretrain_model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda step: 0.5 * (1 + math.cos(math.pi * step / total_steps)))

    trainer.main(args)
    

if __name__ == '__main__':
    args = arg_parse()

    # debug
    # args.pretrain = False
    # args.debug = True
    args.study_type = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
    # args.model_type = 'densenet'

    args.data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    args.teacher_epochs = 1
    
    label_ratio = 0.1
    train_parts_separately(label_ratio, title=f'train_MURA_{int(label_ratio * 100)}%_data_epoch_{args.teacher_epochs}_')
    
    # num_label = 1000
    # train_all_parts(args, num_label=num_label, title=f'train_MURA_{num_label}_all_parts_rbf_epoch_{args.teacher_epochs}_')

