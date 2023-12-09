# -*- coding: utf-8 -*-
import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.semi_supervised import LabelPropagation

from Model import Network
from dataset import Pseudo_data, Concat_Pseudo_label_data
from utils import vis_pseudo_data_images, get_original_dataset, vis_LP_pseudo_label


def Label_Propagation(args, ext_feature, all_label, num_labeled_samples):
    # make feature to 2d: (n_samples, n_features)
    ext_feature = ext_feature.reshape((ext_feature.shape[0], -1))
    
    # scaler = StandardScaler()
    # ext_feature = scaler.fit_transform(ext_feature)

    all_indices = np.arange(len(ext_feature))
    unlabeled_indices = all_indices[num_labeled_samples:]
    
    args.logger.info(f'Doing Label_Propagation, total_samples: {len(all_label)}, unlabeled_points: {len(unlabeled_indices)}')
    
    # labeled unlabel_data as -1
    y_train = np.copy(all_label)
    y_train[unlabeled_indices] = -1
    
    # Learn with LabelSpreading
    lp_model = LabelPropagation(kernel='knn') # used rbf or knn
    lp_model.fit(ext_feature, y_train)

    return lp_model.label_distributions_[unlabeled_indices]


def get_feature_and_label_propagation(args, teacher_model, model_type, labeled_loader, unlabeled_loader):
    
    features_data_path = 'features_data'
    os.makedirs(features_data_path, exist_ok=True)
    
    args.logger.info('labeled data: predict label and get feature')
    labeled_data_features, labeled_data_gts, labeled_data_softmax_preds, labeled_idx, labeled_data_acc = predict(
        args=args,
        model=teacher_model, 
        model_type=model_type, 
        test_loader=labeled_loader
    )

    # used mmap_mode to avoid OOM
    args.logger.info('saving "labeled_data_features.npy')
    np.save(os.path.join(features_data_path, 'labeled_data_features.npy'), labeled_data_features)
    del labeled_data_features
    labeled_data_features = np.load(os.path.join(features_data_path, 'labeled_data_features.npy'), mmap_mode='r')

    args.logger.info('unlabeled data: predict label and get feature')
    unlabeled_data_features, unlabeled_data_gts, teacher_Pseudo_label, unlabeled_idx, unlabeled_data_acc = predict(
        args=args,
        model=teacher_model,
        model_type=model_type,
        test_loader=unlabeled_loader
    )

    # used mmap_mode to avoid OOM
    args.logger.info('saving "unlabeled_data_features.npy')
    np.save(os.path.join(features_data_path, 'unlabeled_data_features.npy'), unlabeled_data_features)
    del unlabeled_data_features
    unlabeled_data_features = np.load(os.path.join(features_data_path, 'unlabeled_data_features.npy'), mmap_mode='r')
    
    ext_feature = np.concatenate([labeled_data_features, unlabeled_data_features])
    all_label = np.concatenate([labeled_data_gts, unlabeled_data_gts])
    args.logger.info(f'shape: ext_feature:{np.shape(ext_feature)}, all_label: {np.shape(all_label)}')
    
    num_labeled_samples=len(labeled_data_features)
    
    del labeled_data_features, labeled_data_gts, labeled_data_softmax_preds, labeled_data_acc
    del unlabeled_data_features, unlabeled_data_acc
    
    LP_pseudo_labels = Label_Propagation(
        args=args,
        ext_feature=ext_feature, 
        all_label=all_label, 
        num_labeled_samples=num_labeled_samples
    )
    
    vis_LP_pseudo_label(
        data_loader=unlabeled_loader,
        sample_idx=unlabeled_idx,
        LP_labels=LP_pseudo_labels,
        batch_size=5
    )
    
    np.save(os.path.join(features_data_path, 'Label_Propagation_features.npy'), LP_pseudo_labels)
    
    del ext_feature, all_label

    # concat LP's psudeo labels & teacher's Pseudo labels 
    feature = np.concatenate([LP_pseudo_labels, teacher_Pseudo_label], axis=1)
    args.logger.info(f'feature: {np.shape(feature)}, unlabeled_data_gts: {np.shape(unlabeled_data_gts)}')
    feature_data = Concat_Pseudo_label_data(feature, unlabeled_data_gts)
    fc_input_loader = DataLoader(dataset=feature_data, batch_size=args.batch_size)
    
    return fc_input_loader, unlabeled_idx


def predict(args, model, model_type, test_loader):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader)
        correct, total_sample = 0, len(test_loader.dataset)
        pre_soft = np.zeros((total_sample, args.num_classes), dtype=np.float32)
        gt_label = np.zeros(total_sample, dtype=np.int64)
        data_idx = np.zeros(total_sample, dtype=np.int64)
        features = None
        batch_start, batch_end = 0, 0
        
        
        def hook(module, input):
            nonlocal features, batch_start, batch_end
            input = input[0]
            if features is None:
                features = np.zeros((total_sample,) + input.shape[1:], dtype=np.float32)
                
            batch_end = batch_start + input.shape[0]
            features[batch_start : batch_end, ...] = input.detach().cpu().numpy()
            
            
        # need to rewrite feature extract for different models
        if model_type == 'resnet':
            handle = model.model.fc.register_forward_pre_hook(hook)
            
        for batch in pbar:
            
            imgs, labels = batch['img'].to(args.device), batch['label'].to(args.device)

            out = model(imgs)
            predict_softmax = F.softmax(out, dim=1)
    
            _, pre = torch.max(out.data, 1)
            
            correct += (pre == labels).sum().item()
            
            gt_label[batch_start : batch_end, ...] = labels.cpu()
            pre_soft[batch_start : batch_end, ...] = predict_softmax.cpu()
            data_idx[batch_start : batch_end] = batch['idx']
            batch_start += args.batch_size
        
        handle.remove()
    
    
    return features, gt_label, pre_soft, data_idx, correct / len(test_loader.dataset)


def train_FC(args, data_loader_train, model, device, save_dir):
    
    os.makedirs(save_dir,exist_ok=True)
    
    model.train()
    model.to(device)
    model = model.float()
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    all = []
    
    for epoch in range(1, args.FC_epochs + 1):
        running_loss = 0.0
        running_correct = 0
        pbar = tqdm(data_loader_train)
        cnt = 0
        
        for step, data in enumerate(pbar):
            pbar.set_description(f'[epoch {epoch}/{args.FC_epochs}]')
            X_train, y_train = data
            cnt += len(X_train)
            X_train, y_train = Variable(X_train), Variable(y_train)
            
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            inputs = model(X_train)
            _, pred = torch.max(inputs.data, 1)
            optimizer.zero_grad()
            
            loss = cost(inputs, y_train)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            running_correct += torch.sum(pred == y_train.data)
            pbar.set_postfix(Loss=running_loss.item()/(cnt), correct=running_correct.item() / (cnt))

        # args.logger.info(
        #     "[epoch {}/{}]Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(
        #         epoch, (args.FC_epochs), running_loss.item()/len(data_loader_train.dataset), 100 * running_correct.item() / len(data_loader_train.dataset),
        #         )
        # )

        epoch_acc = running_correct.double() / len(data_loader_train.dataset)
        all.append(running_loss.item() / len(data_loader_train.dataset))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    
    fig_path = os.path.join('pic', 'FC')
    os.makedirs(fig_path, exist_ok=True)
        
    plt.figure(figsize=(7,7),dpi=200)
    plt.plot(all, 'b-o')
    plt.title("Training Curve")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(fig_path,'FC_train_loss.png'))
    plt.close()

    time_since = time.time() - since
    args.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    args.logger.info('Pseudo label prediction model Best Acc: {:4f}'.format(best_acc))
    
    torch.save(model, os.path.join(save_dir, 'last.pt'))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(save_dir, 'best.pt'))

    return model


def test_FC(args, data_loader, model_fc):
    
    model_fc.to(args.device)
    model_fc.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        testing_correct = 0
        pre_soft=[]
        pre_hard=[]
        pbar = tqdm(data_loader)
        for imgs, labels in pbar:
            
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            
            out = model_fc(imgs)
            
            predict_softmax = F.softmax(out, dim=1)
        
            _, pre = torch.max(out.data, 1)
            
            total += labels.size(0)
            correct += (pre == labels).sum().item()
            pre_hard.append(pre.cpu())
            pre_soft.append(predict_softmax.cpu())
            
        args.logger.info('pesudo label predict model Accuracy: {}'.format(correct / total))

        return torch.cat(pre_soft, dim=0), torch.cat(pre_hard, dim=0)


def train(args, iteration, n_epochs, model, data_loader_train, data_loader_val, num_classes, optimizer, Criterion, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    img_dir = 'pic'
    os.makedirs(img_dir, exist_ok=True)

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
            label = data['label']
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

            X_train, label, omega = data['img'], data['label'], data['omega']

            cnt += len(X_train)
            X_train, y_train = Variable(X_train), Variable(label)
            
            X_train = X_train.to(args.device)
            y_train = y_train.to(args.device)
            omega = omega.to(args.device)
            z = torch.index_select(zeta, 0, label).to(args.device)

            inputs = model(X_train)
            _, pred = torch.max(inputs.data, 1)
            optimizer.zero_grad()
            loss = Criterion(inputs, y_train).sum()
            # loss = (Criterion(inputs, y_train) * omega * (1 / z)).sum()
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
            X_val, y_val = data['img'], data['label']

            val_cnt += len(X_val)
            X_val, y_val = Variable(X_val), Variable(y_val)
            X_val = X_val.to(args.device)
            y_val = y_val.to(args.device)
            inputs = model(X_val)
            _, pred = torch.max(inputs.data, 1)
            loss = Criterion(inputs, y_val).mean()
            val_loss += loss.item()
            val_correct += torch.sum(pred == y_val.data).item()
            pbar_val.set_postfix(Loss=val_loss / val_cnt, correct=val_correct / val_cnt)
        
        args.logger.info("[epoch {}/{}]Train Loss is:{:.8f},valid Loss is:{:.8f}, Train Accuracy is:{:.4f}%, valid Accuracy is:{:.4f}%"
                .format(epoch,(n_epochs),
                    train_loss / cnt,
                    val_loss / val_cnt,
                    100 * running_correct / cnt,
                    100 * val_correct / val_cnt,
                    )
                )

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
    plt.close()
    
    time_since = time.time() - since
    args.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_since // 60, time_since % 60))
    args.logger.info('save model at epoch: {}, Best val Acc: {:4f}'.format(best_epoch,best_acc))

    torch.save(model, os.path.join(save_dir, 'last.pt'))

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(save_dir, 'best.pt'))

    return model, best_acc
    
   
def add_pseudo(args, pre_soft, pre_hard, dataset_type, labeled_data, full_data, unlabeled_sample_idx, num_classes, threshold=0.9):
    
    pseudo_data_list = []
    unlabeled_data_list = []
    
    if dataset_type == 'MNIST':
        for i in trange(len(pre_soft),total=len(pre_soft), desc='add_pseudo:'):
            batch = full_data[unlabeled_sample_idx[i]]
            
            if max(pre_soft[i]) >= threshold:
                batch['omega'] = 1 - (max(pre_soft[i]) / np.log(num_classes))
                batch['label'] = pre_hard[i]
                pseudo_data_list.append(batch)
            else:
                unlabeled_data_list.append(batch)
                
        # create new data
        pseudo_data = Pseudo_data(pseudo_data_list)
        vis_pseudo_data_images(
            torch.utils.data.DataLoader(pseudo_data, batch_size=args.batch_size),
            vis_batch_num=5, 
            log_dir=os.path.join('log', 'pseudo_labels')
        )
        
        new_train_data = labeled_data + pseudo_data # return: ConcatDataset
        new_train_loader = torch.utils.data.DataLoader(
            dataset=new_train_data,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=8
        )
        
        new_unlabel_data = Pseudo_data(unlabeled_data_list)
        new_unlabeled_loader = torch.utils.data.DataLoader(
            dataset=new_unlabel_data,
            batch_size=args.batch_size,
            shuffle=len(new_unlabel_data) > 0, # Set shuffle to False if no new unlabeled data
            num_workers=8
        )
        
    args.logger.info(f'Add {len(pseudo_data)} pseudo labels into training set, num_labeled_data: {len(new_train_data)}, num_unlabeled_data: {len(new_unlabel_data)}')
        
    return new_train_loader, new_unlabeled_loader


def fine_tune_pretrain_model(args, model, model_fc, train_loader, val_loader, num_classes, saved_dir, epochs=10):

    # pretrain teacher model
    model = model.to(args.device)
    Criterion = nn.CrossEntropyLoss(reduction='none')
    Criterion = Criterion.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    teacher_model, teacher_acc = train(
        args=args,
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

    args.logger.info(f'fine-tune pre-train model finished, save parameter at {saved_dir}')

    # pretrain pseudo label predict model(FC)
    args.logger.info('train pseudo label predict model(FC)...')
    model_fc.train()

    fc_input_loader, unlabeled_sample_idx = get_feature_and_label_propagation(
        args=args,
        teacher_model=teacher_model,
        model_type=args.model_type,
        labeled_loader=train_loader,
        unlabeled_loader=val_loader
    )
    
    # train FC
    model_fc = train_FC(
        args=args,
        data_loader_train=fc_input_loader,
        model=model_fc,
        device=args.device,
        save_dir=os.path.join(saved_dir, 'FC')
    )

    return teacher_model, model_fc
    

def self_training_cycle(args, iteration, teacher_model, pseudo_label_model, train_loader, val_loader, unlabeled_loader, num_classes, save_model_dir):

    teacher_model = teacher_model.to(args.device)

    Criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    optimizer = torch.optim.Adam(teacher_model.parameters(), 0.001)

    fc_input_loader, unlabeled_sample_idx = get_feature_and_label_propagation(
        args=args,
        teacher_model=teacher_model,
        model_type=args.model_type,
        labeled_loader=train_loader,
        unlabeled_loader=unlabeled_loader
    )

    args.logger.info('use model and LB predict label to predict pesudo label...')
    pre_soft, pre_hard = test_FC(
        args=args,
        data_loader=fc_input_loader,
        model_fc=pseudo_label_model,
    )
    
    # add pseudo-label data to labeled data
    train_loader, unlabeled_loader = add_pseudo(
        args=args,
        pre_soft=pre_soft,
        pre_hard=pre_hard,
        dataset_type='MNIST',
        labeled_data=train_loader.dataset, # train_loader need to concat pseudo data, so call concatDataset
        full_data=get_original_dataset(unlabeled_loader.dataset), # call original dataset to ensure idx
        unlabeled_sample_idx=unlabeled_sample_idx,
        num_classes=num_classes
    )

    student_model, student_acc = train(
        args=args,
        iteration=iteration,
        n_epochs=args.student_epochs,
        model=teacher_model,
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        num_classes=num_classes,
        optimizer=optimizer,
        Criterion=Criterion,
        save_dir=os.path.join(save_model_dir, f'student_cycle_{iteration}')
    )

    return student_model, train_loader, unlabeled_loader, student_acc


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
    args.logger.info(f'Self-training completed, best student acc: {best_acc}')
    plt.figure(figsize=(7, 7), dpi=200)
    plt.plot(student_test_acc, 'b-o', label='test_curve')
    plt.title("test acc Curve")
    plt.xlabel("cycle")
    plt.ylabel("acc")
    plt.savefig(os.path.join('pic', 'student_test_acc_curve.png'))
    plt.close()