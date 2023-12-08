# -*- coding: utf-8 -*-
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import random_split

from Model import resnet50, FC
from dataset import ISIC2018_Dataset

from opt import arg_parse
from utils import get_logger
from train_utils import fine_tune_pretrain_model, self_training_cycle


def main(args):
    
    os.makedirs(args.save_model_dir, exist_ok=True)
    
    if args.pretrain:
        args.logger.info('fin-tune pre-train model...')
        args.logger.info(f'===cycle: {0}, labeled data: {len(args.train_loader.dataset)}, unlabeled data: {len(args.unlabeled_loader.dataset)}=======')
        
        teacher_model, pseudo_label_model = fine_tune_pretrain_model(
            args=args,
            model=args.pretrain_model,
            model_fc=args.model_fc, 
            train_loader=args.train_loader,
            val_loader=args.val_loader,
            num_classes=args.num_classes,
            saved_dir=os.path.join(args.save_model_dir, 'pretrain'),
            epochs=args.teacher_epochs
        )
    else:
        args.logger.info('Load pre-train model...')
        teacher_model = torch.load(os.path.join(args.save_model_dir, 'pretrain', 'best.pt'))
        pseudo_label_model = torch.load(os.path.join(args.save_model_dir, 'pretrain', 'FC', 'best.pt'))

    student_test_acc = []
    best_acc = 0
    
    # Self Training pipeline
    for iteration in range(1, args.max_self_training_iteration + 1):
        args.logger.info(f'===Self Training Cycle: {iteration}, labeled data: {len(args.train_loader.dataset)}, unlabeled data: {len(args.unlabeled_loader.dataset)}=======')

        student_model, args.train_loader, args.unlabeled_loader, student_acc = self_training_cycle(
            args=args,
            iteration=iteration, 
            teacher_model=teacher_model,
            pseudo_label_model=pseudo_label_model, 
            train_loader=args.train_loader, 
            val_loader=args.val_loader, 
            unlabeled_loader=args.unlabeled_loader,
            num_classes=args.num_classes,
            save_model_dir=args.save_model_dir
        )
        
        args.logger.info(f'Self-training cycle: {iteration}, student acc: {student_acc}, num_unlabeled_data: {len(args.unlabeled_loader.dataset)}')
        student_test_acc.append(student_acc)
        
        if student_acc > best_acc:
            best_acc = student_acc
            torch.save(student_model, os.path.join(args.save_model_dir, f'best_student_cycle_{iteration}.pt'))
        
        # stop self-training if no unlabeled data
        if len(args.unlabeled_loader.dataset) == 0:
            args.logger.info(f'Self-training completed due to no unlabeled data at cycle: {iteration}')
            break
        else:
            teacher_model = student_model

    # plt self-training test curve
    plt.figure(figsize=(7, 7), dpi=200)
    plt.plot(student_test_acc, 'b-o', label='test_curve')
    plt.title("test acc Curve")
    plt.xlabel("cycle")
    plt.ylabel("acc")
    plt.savefig(os.path.join('pic', 'student_test_acc_curve.png'))
    plt.close()


if __name__ == '__main__':
    args = arg_parse()
    # args.pretrain = False # debug
    args.dataset_dir = os.path.join('Datasets', 'ISIC2018')
    args.save_model_dir = os.path.join('trained_model', 'ISIC2018', 'resnet18')
    args.log_filename = os.path.join('log', 'train_ISIC2018.log')
    
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
    
    main(args)