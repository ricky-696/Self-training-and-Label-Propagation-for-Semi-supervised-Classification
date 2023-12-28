# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import random_split

from Model import resnet18, Same_Label, vgg16
from SRC_MT.code.networks.models import DenseNet121
from dataset import ISIC2018_Dataset, ISIC2018_SRC

import trainer
from opt import arg_parse
from utils import get_logger


if __name__ == '__main__':
    args = arg_parse()
    torch.hub.set_dir('trained_model')

    # debug
    # args.pretrain = False
    # args.debug = True
    args.title = 'ISIC2018_epoch60_20%_affine_res'
    args.model_type = 'resnet'
    args.dataset_dir = os.path.join('Datasets', 'ISIC2018')
    args.teacher_epochs = 60
    args.student_epochs = 20
    args.save_model_dir = os.path.join('trained_model', args.title, args.model_type)
    args.log_filename = f'train_{args.title}'
    args.img_size = (224, 224)
    
    args.logger = get_logger(args.log_filename)
    args.device = torch.device(f'cuda:{args.devices[0]}' if torch.cuda.is_available() else 'cpu')
    args.logger.info(f'Start batch size: {args.batch_size}, device: {args.device}')

    os.makedirs('pic', exist_ok=True)

    args.data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(0.1, 0.1),
                scale=(0.5, 2),
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    args.num_classes = 7
    # data_train = ISIC2018_Dataset(type='train', transform=args.data_transforms['train'])
    # labeled_data, unlabeled_data = random_split(data_train, [len(data_train) - 1, 1])

    # val_data = ISIC2018_Dataset(type='valid', transform=args.data_transforms['test'])

    # exp setting on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9095275&tag=1
    train_data = ISIC2018_SRC(type='training', transform=args.data_transforms['train'])
    val_data = ISIC2018_SRC(type='training', transform=args.data_transforms['test'])
    test_data = ISIC2018_SRC(type='training', transform=args.data_transforms['test'])
    labeled_data, unlabeled_data = random_split(train_data, [0.2, 0.8])

    args.train_loader = torch.utils.data.DataLoader(
        dataset=labeled_data,
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
        args.pretrain_model = resnet18(num_classes=args.num_classes).to(args.device)
    elif args.model_type == 'densenet':
        args.pretrain_model = DenseNet121(num_classes=args.num_classes).to(args.device)
    else:
        args.pretrain_model = vgg16(num_classes=args.num_classes).to(args.device)

    args.model_fc = Same_Label(num_classes=args.num_classes, device=args.device).to(args.device)
    
    args.criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    args.optimizer = torch.optim.Adam(args.pretrain_model.parameters(), 0.001)
    trainer.main(args)


    # test
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=args.shuffle,
        num_workers=8
    )

    model = torch.load(os.path.join(args.save_model_dir, 'student_best.pt')).to(args.device)
    features, gt_label, pre_soft, data_idx, acc = trainer.predict(args, model, args.model_type, test_loader)
    args.logger.info(f'ISIC testset acc: {acc}')