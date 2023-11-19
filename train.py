# -*- coding: utf-8 -*-
import os
import copy
import glob
import time
import torch
import random
import logging
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from itertools import count
from collections import Counter
from torch import optim

from torch.autograd import Variable
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler, random_split

from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

from Model import FC, gray_resnet18
from dataset import Psuedo_data, Concat_Psuedo_label_data, MNIST_omega


os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
	'[%(levelname)1.1s %(asctime)s %(funcName)s:%(lineno)d] %(message)s',
	datefmt='%Y%m%d %H:%M:%S')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

log_filename = 'log/Med_SelfTraining.log'
log_dir = os.path.dirname(log_filename)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logging.getLogger('matplotlib.font_manager').disabled = True
logger.addHandler(ch)
logger.addHandler(fh)


def Label_Propagation(ext_feature, all_label, num_labeled_samples):
    # make feature to 2d: (n_samples, n_features)
    ext_feature = ext_feature.reshape((ext_feature.shape[0], -1))
    
    scaler = StandardScaler()
    ext_feature = scaler.fit_transform(ext_feature)

    all_indices = np.arange(len(ext_feature))
    unlabeled_indices = all_indices[num_labeled_samples:]
    
    logger.info(f'LB, total_samples: {len(all_label)}, labeled_points: {len(unlabeled_indices)}')
    
    # labeled unlabel_data as -1
    y_train = np.copy(all_label)
    y_train[unlabeled_indices] = -1
    
    # Learn with LabelSpreading
    lp_model = LabelPropagation(kernel='rbf') # used rbf or knn
    lp_model.fit(ext_feature, y_train)

    return lp_model.label_distributions_, unlabeled_indices


def tensor_to_list(x, k):
    x = x.cpu().numpy().tolist()
    k.extend(x)
    
    return k


def test(model, model_type, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        pre_soft, gt_label, features = [], [], []

        def hook(module, input, output): 
            x = output.detach().cpu()
            features.append(x)
        
        # need to rewrite feature extract for different models
        if model_type == 'resnet':
            handle = model.model.layer4[-1].register_forward_hook(hook)
            
        pbar = tqdm(test_loader)
        for imgs, labels, _ in pbar:
            
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            predict_softmax = F.softmax(out, dim=1)
    
            _, pre = torch.max(out.data, 1)
            
            correct += (pre == labels).sum().item()
            
            gt_label.append(labels.cpu())
            pre_soft.append(predict_softmax.cpu())
    
    handle.remove()

    return torch.cat(features, dim=0), torch.cat(gt_label, dim=0), torch.cat(pre_soft, dim=0), correct / len(test_loader.dataset)


def train_FC(epochs, data_loader_train, model, dataset, device, save_dir):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.train()
    model.to(device)
    model = model.float()
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    all = []
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_correct = 0
        pbar = tqdm(data_loader_train)
        cnt = 0
        
        for step, data in enumerate(pbar):
            pbar.set_description(f'[epoch {epoch}/{epochs}]')
            X_train, y_train = data
            cnt += len(X_train)
            X_train, y_train = Variable(X_train), Variable(y_train)
            
            X_train = X_train.to(device)
            y_train = y_train.to(device)
                
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            
            loss = cost(outputs, y_train)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            running_correct += torch.sum(pred == y_train.data)
            pbar.set_postfix(Loss=running_loss.item()/(cnt), correct=running_correct.item() / (cnt))

        logger.info(
            "[epoch {}/{}]Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(
                epoch, (epochs), running_loss.item()/len(dataset), 100 * running_correct.item() / len(dataset),
                )
        )

        epoch_acc = running_correct.double() / len(dataset)
        all.append(running_loss.item() / len(dataset))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    
    fig_path = os.path.join('pic', 'FC')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        
    plt.figure(figsize=(7,7),dpi=200)
    plt.plot(all, 'b-o')
    plt.title("Training Curve")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(fig_path,'FC_train_loss.png'))
    time_since = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    logger.info('Best Acc: {:4f}'.format(best_acc))
    
    torch.save(model, os.path.join(save_dir, 'last.pt'))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(save_dir, 'best.pt'))

    return model


def test_FC(data_loader, modelfc, device):
    
    modelfc.to(device)
    modelfc.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        testing_correct = 0
        pre_soft=[]
        pre_hard=[]
        pbar = tqdm(data_loader)
        for imgs, labels in pbar:
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            out = modelfc(imgs)
            
            predict_softmax = F.softmax(out, dim=1)
        
            _, pre = torch.max(out.data, 1)
            
            total += labels.size(0)
            correct += (pre == labels).sum().item()
            pre_hard.append(pre.cpu())
            pre_soft.append(predict_softmax.cpu())
            
        logger.info('pesudo label predict model Accuracy: {}'.format(correct / total))

        return torch.cat(pre_soft, dim=0), torch.cat(pre_hard, dim=0)


def train(iteration, n_epochs, model, data_loader_train, data_loader_val, num_classes, optimizer, Criterion, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_dir = 'pic'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    model.train()
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    train_loss_curve=[]
    val_loss_curve=[]
    train_acc_curve=[]
    val_acc_curve=[]

    # loss weight
    if iteration <= 0:
        zeta = np.ones(num_classes)
    else:
        zeta = np.zeros(num_classes)
        for i, data in enumerate(data_loader_train):
            image, label, _ = data
            zeta[label] += 1

    zeta = torch.tensor(zeta)

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        running_correct = 0
        cnt = 0
        step = 0
        pbar_train = tqdm(data_loader_train)
        
        for step, data in enumerate(pbar_train):
            pbar_train.set_description(f'[epoch {epoch}/{n_epochs}][train]')

            X_train, label, omega = data

            cnt += len(X_train)
            X_train, y_train = Variable(X_train), Variable(label)
            
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            omega = omega.to(device)
            z = torch.index_select(zeta, 0, label).to(device)

            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = (Criterion(outputs, y_train) * omega * (1 / z)).sum()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            running_correct += torch.sum(pred == y_train.data).item()

            pbar_train.set_postfix(
                Loss=train_loss / cnt,
                correct=running_correct / cnt
            )

        val_loss = 0.0
        val_correct = 0
        val_cnt = 0
        pbar_val = tqdm(data_loader_val)

        for data in pbar_val:
            pbar_val.set_description(f'[epoch {epoch}/{n_epochs}][val]')
            X_val, y_val,_ = data

            val_cnt += len(X_val)
            X_val, y_val = Variable(X_val), Variable(y_val)
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs = model(X_val)
            _, pred = torch.max(outputs.data, 1)
            loss = Criterion(outputs, y_val).mean()
            val_loss += loss.item()
            val_correct += torch.sum(pred == y_val.data).item()
            pbar_val.set_postfix(Loss=val_loss / val_cnt, correct=val_correct / val_cnt)
        
        logger.info("[epoch {}/{}]Train Loss is:{:.8f},valid Loss is:{:.8f}, Train Accuracy is:{:.4f}%, valid Accuracy is:{:.4f}%"
                .format(epoch,(n_epochs),
                    train_loss / cnt,
                    val_loss / val_cnt,
                    100 * running_correct / cnt,
                    100 * val_correct / val_cnt,
                    )
                )

        # epoch_acc = running_correct.double()/cnt
        epoch_acc = val_correct / val_cnt
        train_loss_curve.append(train_loss / cnt)
        val_loss_curve.append(val_loss / val_cnt)
        train_acc_curve.append(running_correct / cnt)
        val_acc_curve.append(val_correct / val_cnt)

        if epoch_acc >= best_acc:
            best_epoch, best_acc = epoch, epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())


    plt.figure(figsize=(7, 7),dpi=200)
    plt.plot(train_loss_curve, 'r-o', label='train_curve')
    plt.plot(val_loss_curve,'b-o',label='val_curve')
    plt.legend()
    plt.title("training Loss Curve")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(img_dir, 'iteration_' + str(iteration) + '_loss.png'))
    plt.figure(figsize=(7,7),dpi=200)
    plt.plot(train_acc_curve,'r-o',label='train_curve')
    plt.plot(val_acc_curve,'b-o',label='val_curve')
    plt.legend()
    plt.title("training acc Curve")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.savefig(os.path.join(img_dir, 'iteration_' + str(iteration) + '_acc.png'))
    time_since = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_since // 60, time_since % 60))
    logger.info('save model at epoch: {}, Best val Acc: {:4f}'.format(best_epoch,best_acc))

    torch.save(model, os.path.join(save_dir, 'last.pt'))

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(save_dir, 'best.pt'))

    return model, best_acc
    
def add_pseudo(pre_soft, pre_hard, dataset_type, train_data, unlabeled_loader, num_classes, threshold=0.9):
    
    # get the idx for unlabeled data
    unlabeled_sample_idx = list(unlabeled_loader.sampler)
    pseudo_data_list = []
    unlabeled_data_list = []
    
    if dataset_type == 'MNIST':
        for i in range(len(pre_soft)):
            img, label, omega = unlabeled_loader.dataset[unlabeled_sample_idx[i]]
            
            if max(pre_soft[i]) >= threshold:
                omega = 1 - (max(pre_soft[i]) / np.log(num_classes))
                pseudo_data_list.append((img, pre_hard[i], omega))
            else:
                unlabeled_data_list.append((img, label, omega))
                
        # create new data
        pseudo_data = Psuedo_data(pseudo_data_list)
        new_train_data = train_data + pseudo_data # return: ConcatDataset
        new_unlabel_data = Psuedo_data(unlabeled_data_list)
        
        new_train_loader = torch.utils.data.DataLoader(
            dataset=new_train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8
        )
        
        new_unlabeled_loader = torch.utils.data.DataLoader(
            dataset=new_unlabel_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8
        )
        
    return new_train_loader, new_unlabeled_loader


def fine_tune_pretrain_model(model, train_loader, val_loader, num_classes, saved_dir, epochs=10):

    # pretrain teacher model
    model = model.to(device)
    Criterion = nn.CrossEntropyLoss(reduction='none')
    Criterion = Criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(),0.001)

    teacher_model, teacher_acc = train(
        iteration=0,
        n_epochs=epochs,
        model=model,
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        num_classes=num_classes,
        optimizer=optimizer,
        Criterion=Criterion,
        save_dir=saved_dir
    )

    logger.info(f'fine-tune pre-train model finished, save parameter at {saved_dir}')

    # pretrain pseudo label predict model(FC)
    logger.info('train pseudo label predict model(FC)...')
    modelfc = FC(num_classes=int(num_classes * 2))
    modelfc.to(device)
    modelfc.train()

    logger.info('labeled data: predict label and get feature')
    labeled_data_features, labeled_data_gts, labeled_data_softmax_preds, labeled_data_acc = test( # 對label data做特徵擷取
        model=teacher_model,
        model_type='resnet',
        test_loader=train_loader,
        device=device
    )

    logger.info('unlabeled FC data: predict label and get feature')
    unlabeled_data_features, unlabeled_data_gts, unlabeled_data_softmax_preds, unlabeled_data_acc = test( # 對unlabeled fc data做特徵擷取
        model=teacher_model,
        model_type='resnet',
        test_loader=val_loader,
        device=device
    )

    predict_softmax = np.array(unlabeled_data_softmax_preds)
    ext_feature = np.concatenate([labeled_data_features, unlabeled_data_features])
    all_label = np.concatenate([labeled_data_gts, unlabeled_data_gts])
    logger.info(f'shape: ext_feature:{np.shape(ext_feature)}, all_label: {np.shape(all_label)}')
    
    lb_out, idx = Label_Propagation(
        ext_feature=ext_feature, 
        all_label=all_label, 
        num_labeled_samples=len(labeled_data_features)
    )

    # concat LP's psudeo labels & teacher's psuedo labels 
    feature = np.concatenate((lb_out[idx], predict_softmax), axis=1)
    logger.info(f'feature: {np.shape(feature)}, unlabeled_data_gts: {np.shape(unlabeled_data_gts)}')
    feature_data = Concat_Psuedo_label_data(feature, unlabeled_data_gts)
    fc_input_loader = DataLoader(dataset=feature_data, batch_size=batch_size)
    
    # train FC
    modelfc = train_FC(
        epochs=100,
        data_loader_train=fc_input_loader,
        model=modelfc,
        dataset=feature_data,
        device=device,
        save_dir=os.path.join(saved_dir, 'FC')
    )

    return teacher_model, modelfc
    

def self_training_cycle(iteration, teacher_model, pseudo_label_model, train_loader, val_loader, unlabeled_loader, num_classes, save_model_dir):

    teacher_model = teacher_model.to(device)

    Criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(teacher_model.parameters(), 0.001)

    logger.info('labeled data: predict label and get feature')
    labeled_data_features, labeled_data_gts, labeled_data_softmax_preds, labeled_data_acc = test( # 對label data做特徵擷取
        model=teacher_model,
        model_type='resnet',
        test_loader=train_loader,
        device=device
    )

    logger.info('unlabeled FC data: predict label and get feature')
    unlabeled_data_features, unlabeled_data_gts, unlabeled_data_softmax_preds, unlabeled_data_acc = test( # 對unlabeled fc data做特徵擷取
        model=teacher_model,
        model_type='resnet',
        test_loader=unlabeled_loader,
        device=device
    )
    
    predict_softmax = np.array(unlabeled_data_softmax_preds)
    ext_feature = np.concatenate([labeled_data_features, unlabeled_data_features])
    all_label = np.concatenate([labeled_data_gts, unlabeled_data_gts])
    logger.info(f'Do label propagation: ext_feature:{np.shape(ext_feature)}, all_label: {np.shape(all_label)}')
    
    lb_out, idx = Label_Propagation(
        ext_feature=ext_feature, 
        all_label=all_label, 
        num_labeled_samples=len(labeled_data_features)
    )

    # concat LP's psudeo labels & teacher's psuedo labels 
    feature = np.concatenate((lb_out[idx], predict_softmax), axis=1)
    logger.info(f'feature: {np.shape(feature)}, unlabeled_data_gts: {np.shape(unlabeled_data_gts)}')
    feature_data = Concat_Psuedo_label_data(feature, unlabeled_data_gts)
    fc_input_loader = DataLoader(dataset=feature_data, batch_size=batch_size)

    logger.info('use model and LB predict label to predict pesudo label...')
    pre_soft, pre_hard = test_FC(
        fc_input_loader,
        pseudo_label_model,
        device
    )
    
    # add pseudo-label data to labeled data
    train_loader, unlabeled_loader = add_pseudo(
        pre_soft=pre_soft,
        pre_hard=pre_hard,
        dataset_type='MNIST',
        train_data=train_loader.dataset,
        unlabeled_loader=unlabeled_loader,
        num_classes=num_classes
    )

    student_model, student_acc = train(
        iteration=iteration,
        n_epochs=student_epochs,
        model=teacher_model,
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        num_classes=num_classes,
        optimizer=optimizer,
        Criterion=Criterion,
        save_dir=os.path.join(save_model_dir, f'student_cycle_{iteration}')
    )

    return student_model, train_loader, unlabeled_loader, student_acc


def main(train_loader, val_loader, unlabeled_loader, pretrain_model, save_model_dir):
    
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    logger.info('fin-tune pre-train model...')
    logger.info(f'===cycle: {0}, labeled data: {len(train_loader.dataset)}, unlabeled data: {len(unlabeled_loader.dataset)}=======')
    
    teacher_model, pseudo_label_model = fine_tune_pretrain_model(
        model=pretrain_model, 
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        saved_dir=os.path.join(save_model_dir, 'pretrain'),
        epochs=teacher_epochs
    )

    student_test_acc = []
    best_acc = 0
    
    # Self Training pipeline
    for iteration in range(1, max_self_training_iteration + 1):
        logger.info(f'===Self Training Cycle: {iteration}, labeled data: {len(train_loader.dataset)}, unlabeled data: {len(unlabeled_loader.dataset)}=======')

        student_model, train_loader, unlabeled_loader, student_acc = self_training_cycle(
            iteration=iteration, 
            teacher_model=teacher_model,
            pseudo_label_model=pseudo_label_model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            unlabeled_loader=unlabeled_loader,
            num_classes=num_classes,
            save_model_dir=save_model_dir
        )
        
        logger.info(f'Self-training cycle: {iteration}, student acc: {student_acc}, num_unlabeled_data: {len(unlabeled_loader.dataset)}')
        student_test_acc.append(student_acc)
        
        if student_acc > best_acc:
            best_acc = student_acc
            torch.save(student_model, os.path.join(save_model_dir, f'best_student_cycle_{iteration}.pt'))
        
        # stop self-training if no unlabeled data
        if len(unlabeled_loader.dataset) == 0:
            logger.info(f'Self-training completed due to no unlabeled data at cycle: {iteration}')
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


if __name__ == '__main__':
    # ToDo: used args
    batch_size = 32
    teacher_epochs, student_epochs = 2, 10
    debug = False # for debug
    shuffle = not debug
    max_self_training_iteration = 10
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Start batch size: {batch_size}, device: {device}')

    if not os.path.exists('pic'):
        os.makedirs('pic')

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    data_train = MNIST_omega(
        root='./mnist/',
        train=True,
        transform=transform,
        download=True,
        debug=debug
    )  
    
    data_test = MNIST_omega(
        root='./mnist/',
        train=False,
        transform=transform,
        debug=debug
    )

    num_classes = data_train.get_num_classes()

    if debug:
        labeled_data, unlabeled_data = data_train, data_test
        train_data, val_data = data_train, data_train
    else:
        train_data, unlabeled_data = random_split(data_train, [0.5, 0.5])
        val_data = data_test

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8
    )
    
    unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8
    )
    
    pretrain_model = gray_resnet18(num_classes=num_classes)
    

    # # debug
    # train_loader, unlabeled_loader = add_pseudo(
    #     pre_soft=[[0.91, 0.8, 0.7], [0.8, 0.91, 0.7], [0.7, 0.8, 0.87]],
    #     pre_hard=[0, 1, 2],
    #     dataset_type='MNIST',
    #     train_data=train_loader.dataset,
    #     unlabeled_loader=unlabeled_loader,
    #     num_classes=3
    # )
    
    # for step, data in enumerate(train_loader):
    #     X_train, label, omega = data
    
    main(
        train_loader, 
        val_loader, 
        unlabeled_loader, 
        pretrain_model,
        save_model_dir=os.path.join('trained_model', 'resnet18')
    )
