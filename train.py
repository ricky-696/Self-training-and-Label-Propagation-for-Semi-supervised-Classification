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
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.semi_supervised import LabelPropagation

from Model import FC, gray_resnet18
from dataset import ISIC_Dataset, MyDataset, MNIST_omega


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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def select_balance_data(dataset:ISIC_Dataset, train_idx, n_test=70):
    labels = np.array(dataset.target)[train_idx]
    classes = np.unique(labels)

    ixs = []
    for cl in classes:
        ixs.append(np.random.choice(np.nonzero(labels==cl)[0], n_test,
                replace=False))

    # take same num of samples from all classes
    # ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    # ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])
    ix_unlabel = train_idx[ixs]
    ix_label = train_idx[not ixs]

    return ix_label, ix_unlabel


def save_object(obj, filename):
    try:
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(f"{filename}.pickle", "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def LB(ext_feature, all_label, labeled_data):
    
    # print(type(ext_feature))
    digits = datasets.load_digits()
    rng = np.random.RandomState(0)
    indices = np.arange(len(digits.data))
    rng.shuffle(indices)
    
    n_total_samples = len(all_label) ##
    n_labeled_points = len(labeled_data) ##
    # print('LB:',n_total_samples,n_labeled_points)
    logger.info(f'LB, total_samples: {n_total_samples}, labeled_points: {n_labeled_points}')
    
    indices = np.arange(n_total_samples)
    
    unlabeled_set = indices[n_labeled_points:]
    
    # #############################################################################
    # Shuffle everything around
    y_train = np.copy(all_label) ##
    y_train[unlabeled_set] = -1
    
    # #############################################################################
    # Learn with LabelSpreading
    lp_model = LabelPropagation(gamma=0.25, max_iter=20)
    lp_model.fit(ext_feature, y_train) ##
    predicted_labels = lp_model.transduction_[unlabeled_set]
    lb_out = lp_model.label_distributions_
    logger.info(f'{lb_out[0]}, {lp_model.label_distributions_[0]}, {len(lb_out)}, {len(lp_model.label_distributions_)}')
    # true_labels = all_label[unlabeled_set] ##
    
    # cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)
    
    # logger.info("Label Spreading model: %d labeled & %d unlabeled points (%d total)" %
    #       (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))
    
    # logger.info(classification_report(true_labels, predicted_labels))
    
    # logger.info("Confusion matrix")
    # logger.info(cm)
    
    # #############################################################################
    # Calculate uncertainty values for each transduced distribution
    #pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
    
    # #############################################################################
    # Pick the top 10 most uncertain labels
   # uncertainty_index = np.argsort(pred_entropies)[-10:]
    return lb_out, unlabeled_set
    # #############################################################################
    # Plot
    '''
    f = plt.figure(figsize=(7, 5))
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]
    
        sub = f.add_subplot(2, 5, index + 1)
        sub.imshow(image, cmap=plt.cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        sub.set_title('predict: %i\ntrue: %i' % (
            lp_model.transduction_[image_index], y[image_index]))
    
    f.suptitle('Learning with small amount of labeled data')
    plt.show()
    '''


def toarray_add(x, k):
    x = x.cpu().numpy()
    x = x.tolist()
    for i in range(len(x)):
        k.append(x[i])

    return k


def test(model, test_loader, device, iteration):
    with torch.no_grad():
        correct = 0
        total = 0
        testing_correct = 0
        #all=[]
        pre_soft = []
        all_label = np.arange(0)
        features = []

        def hook(module, input, output): 
            x = output.clone().detach().cpu().numpy()
            x = x.tolist()
            for i in range(len(x)):
                features.append(x[i])
        # features.append(output.clone().detach())
        
        pbar = tqdm(test_loader)
        for imgs, labels, _ in pbar:
            
            imgs, labels = imgs.to(device), labels.to(device)

            # need to rewrite feature extract for different models
            handle = model.model.layer4[-1].register_forward_hook(hook)

            out = model(imgs)
            handle.remove()

            predict_softmax = F.softmax(out)
            
            _, pre = torch.max(out.data, 1)
            
            total += labels.size(0)
            correct += (pre == labels).sum().item()
            labels = labels.cpu().numpy()
            
            all_label = np.append(all_label,labels)
            pre_soft = toarray_add(predict_softmax,pre_soft)

            break
     
    if total == 0:
        total = 1

    return features, model, all_label, pre_soft, correct / total


def fullyconect(epochs, data_loader_train, model, dataset, device, tag):
    model.to(device)
    model = model.float()
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    all = []
    for epoch in range(1,epochs+1):
        running_loss = 0.0
        running_correct = 0
        pbar = tqdm(data_loader_train)
        cnt = 0
        for step,data in enumerate(pbar):
            pbar.set_description(f'[epoch {epoch}/{epochs}]')
            X_train, y_train = data
            cnt += len(X_train)
            X_train, y_train = Variable(X_train), Variable(y_train)
            
            X_train = X_train.cuda()
            y_train = y_train.cuda()
                
            outputs = model(X_train.float())
            _,pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            
            loss = cost(outputs, y_train.long())
            
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
        all.append(running_loss / len(dataset))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    plt.figure(figsize=(7,7),dpi=200)
    plt.plot(all,'b-o') #劃出loss曲線
    plt.title("Training Curve")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.savefig(r'C:\Users\Rayeh\Desktop\Med_self-training_Yuan\picture\pretrain'+str(tag)+'_train_loss.png')
    time_since = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    logger.info('Best Acc: {:4f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)

    return model


def fullyconnecttest(data_loader_train, modelfc, device):
    # modelfc = torch.load(r'C:\Users\b3171154\Desktop\碩論\self_training_on_trash\trashnet\mnist_pkl\pre_fctrain_pth\FC50.pth')
    modelfc.to(device)
    modelfc = modelfc.double()
    with torch.no_grad(): # when in test stage, no grad
        correct = 0
        total = 0
        testing_correct = 0
        #all=[]
        pre_soft=[]
        pre_hard=[]
        all_label=np.arange(0)
        pbar = tqdm(data_loader_train)
        for imgs, labels in pbar:
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            out = modelfc(imgs.double())
            
            predict_softmax = F.softmax(modelfc(imgs))#輸出softmax
        
            _,pre = torch.max(out.data, 1) #輸出第一列的最大值跟序號代表預測類別 tensor([5.5664, 7.0837, 5.2682, 4.2807], device='cuda:0') tensor([3, 3, 1, 1], device='cuda:0')
            
            total += labels.size(0)
            correct += (pre == labels).sum().item()
            pre_hard = toarray_add(pre,pre_hard)
            pre_soft = toarray_add(predict_softmax,pre_soft)
        # print(len(pre_soft))
        logger.info('pesudo label predict model Accuracy: {}'.format(correct / total))

        return pre_soft, pre_hard


def train(cycle, n_epochs, model, data_loader_train, data_loader_val, optimizer, Criterion, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_dir = 'pic'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    train_loss_curve=[]
    val_loss_curve=[]
    train_acc_curve=[]
    val_acc_curve=[]
    num_classes = train_loader.dataset.dataset.dataset.get_num_classes()

    # loss weight
    if cycle <= 1:
        zeta = np.ones(num_classes)
    else:
        zeta = np.zeros(num_classes)
        for i, data in enumerate(data_loader_train):
            image, label, _ = data
            zeta[label] += 1

    zeta = torch.tensor(zeta)

    for epoch in range(1,n_epochs+1):
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
            
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            omega = omega.cuda()
            z = torch.index_select(zeta, 0, label).cuda()

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
            break

        val_loss = 0.0
        val_correct = 0
        val_cnt = 0
        pbar_val = tqdm(data_loader_val)

        for data in pbar_val:
            pbar_val.set_description(f'[epoch {epoch}/{n_epochs}][val]')
            X_val, y_val,_ = data

            val_cnt += len(X_val)
            X_val, y_val = Variable(X_val), Variable(y_val)
            X_val = X_val.cuda()
            y_val = y_val.cuda()
            outputs = model(X_val)
            _, pred = torch.max(outputs.data, 1)
            loss = Criterion(outputs, y_val).mean()
            val_loss += loss.item()
            val_correct += torch.sum(pred == y_val.data).item()
            pbar_val.set_postfix(Loss=val_loss / val_cnt, correct=val_correct / val_cnt)
            break
        
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

        break

    plt.figure(figsize=(7, 7),dpi=200)
    plt.plot(train_loss_curve, 'r-o', label='train_curve')
    plt.plot(val_loss_curve,'b-o',label='val_curve')
    plt.legend()
    plt.title("training Loss Curve")
    plt.xlabel("epochs")
    plt.xticks(list(range(1, n_epochs + 1)))
    plt.ylabel("Loss")
    plt.savefig(os.path.join(img_dir, 'cycle_' + str(cycle) + '_loss.png'))
    plt.figure(figsize=(7,7),dpi=200)
    plt.plot(train_acc_curve,'r-o',label='train_curve')
    plt.plot(val_acc_curve,'b-o',label='val_curve')
    plt.legend()
    plt.title("training acc Curve")
    plt.xlabel("epochs")
    plt.xticks(list(range(1, n_epochs + 1)))
    plt.ylabel("acc")
    plt.savefig(os.path.join(img_dir, 'cycle_' + str(cycle) + '_acc.png'))
    time_since = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_since // 60, time_since % 60))
    logger.info('save model at epoch: {}, Best val Acc: {:4f}'.format(best_epoch,best_acc))

    torch.save(model, os.path.join(save_dir, 'last.pt'))

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(save_dir, 'best.pt'))

    return model
    
def addpsudo(pre_soft, pre_hard, label_idx, unlabel_idx, dataset, all_true):
    psudo_num = 0
    rest_num = 0
    wrong_num = 0
    
    # k = math.floor(0.99999999 * (10**(iteration))) / 10**(iteration)
    k = 0.9
    logger.info(f'select high confidience unlabeled data, threshold: {k}, pre_soft: {np.shape(pre_soft)}, unlabel_data: {len(unlabel_idx)}, ture_label: {len(all_true)}')
    remove_list = []
    for i in range(len(pre_soft)):
        if max(pre_soft[i])>= k :
            if pre_hard[i]!=all_true[i]:
                wrong_num += 1
            psudo_num += 1
            label_idx = np.append(label_idx,unlabel_idx[i])
            omega = 1 - (max(pre_soft[i]) / np.log(7))
            dataset.omega[unlabel_idx[i]] = omega
            dataset.target[unlabel_idx[i]] = pre_hard[i]
            remove_list.append(np.where(unlabel_idx==unlabel_idx[i]))
        else:
            rest_num += 1

    unlabel_idx = np.delete(unlabel_idx,remove_list)
    logger.info(f'selected data: {psudo_num}, selected but wrong: {wrong_num}, unlabeled: {rest_num}')
    if psudo_num == 0 and len(pre_soft) > 0:
        logger.info(f'max confidience: {torch.max(torch.tensor(pre_soft))}')

    return label_idx, unlabel_idx, dataset, psudo_num


def fine_tune_pretrain_model(model, train_loader, val_loader, test_loader, saved_dir, model_name, epochs=10):

    # pre task predict model
    model = model.to(device)
    model.train()
    Criterion = nn.CrossEntropyLoss(reduction='none')
    Criterion = Criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(),0.001)

    trained_model = train(
        cycle=0,
        n_epochs=epochs,
        model=model,
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        optimizer=optimizer,
        Criterion=Criterion,
        save_dir=os.path.join(saved_dir, model_name)
    )

    t1, t2, t3, t4, testacc = test(
        model=trained_model,
        test_loader=test_loader, 
        device=device,
        iteration=0
    )

    logger.info(f'teacher model number : {0}, testing acc : {testacc}')
    logger.info(f'fine-tune pre-train model finished, save parameter at {saved_dir}')

    # pretrain train FC
    logger.info('train pesudo label predict model(FC)...')
    modelfc = FC()
    modelfc.to(device)
    modelfc.train()
    logger.info('label data: predict label and get feature')
    ext_feature_1, model_1, all_label_1, predict_softmax_1, t = test(labeled_loader, device, 0)# 對label data 500做特徵擷取
    logger.info('label FC data: predict label and get feature')
    ext_feature_2, model_2, all_label_2, predict_softmax_2, t = test(fc_loader, device,0)# 對fc data 500做特徵擷取
    predict_softmax = predict_softmax_2
    ext_feature = np.concatenate([ext_feature_1,ext_feature_2])
    all_label = np.concatenate([all_label_1,all_label_2])
    logger.info(f'shape: ext_feature:{np.shape(ext_feature)}, all_label: {np.shape(all_label)}')
    
    predict_softmax = np.array(predict_softmax)# 轉成array
    ext_feature = np.array(ext_feature)# 轉成array讓LB做處理
    all_label = np.array(all_label)
    
    lb_out, idx = LB(ext_feature, all_label, labeled_indices)
    feature = np.concatenate((lb_out[idx], predict_softmax), axis=1) # 把兩個預測結果concat起來準備丟進去模型
    logger.info(f'feature: {np.shape(feature)}, all_label_2: {np.shape(all_label_2)}')
    feature_data = MyDataset(feature, all_label_2)
    fc_input_loader = DataLoader(dataset=feature_data,batch_size=batch_size)
    
    # train FC
    modelfc = fullyconect(epochs=100,data_loader_train=fc_input_loader, model=modelfc,dataset=feature_data, device=device, tag='FC')
    torch.save(modelfc,r'C:\Users\Rayeh\Desktop\Med_self-training_Yuan\ISIC2018_500\FC20.pth')
    del modelfc


def self_training_cycle():
    

    return model


if __name__ == '__main__': 
    seed = 0
    set_seed(seed)
    if_use_gpu = 1
    batch_size = 32
    iteration = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Start batch size: {batch_size}, device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    data_train = MNIST_omega(  
        root='./mnist/',
        train=True,
        transform=transform,
        download=True,
    )  
    
    data_test = MNIST_omega(
        root='./mnist/',
        train=False,
        transform=transform,
    )

    num_classes = data_train.get_num_classes()
    labeled_data, unlabeled_data = random_split(data_train, [0.2, 0.8])
    train_data, val_data = random_split(labeled_data, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    
    unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=data_test,
        batch_size=1,
        shuffle=True,
        num_workers=8
    )

    logger.info('fin-tune pre-train model...')
    logger.info(f'===cycle: {0}, labeled data: {len(labeled_data)}, unlabeled data: {len(unlabeled_data)}=======')
    # pre train task model
    finetune_epochs = 10
    pretrain_model = gray_resnet18(num_classes=num_classes)
    
    pretrain_model = fine_tune_pretrain_model(
        model=pretrain_model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        test_loader=test_loader,
        saved_dir='trained_model/pretrain',
        model_name='resnet_18',
        epochs=finetune_epochs
    )

    big = 0
    alltest = []
    patience = 0
    n_epochs = 30
    # true_labeled_data_len = len(labeled_indices)
    
    # pipeline
    while(True):
        iteration += 1
        # check data len
        logger.info(f'===cycle: {iteration}, labeled data: {len(labeled_indices)}, unlabeled data: {len(unlabeled_indices)}=======')

        logger.info('create and train new student model')
        model = gray_resnet18(num_classes=num_classes)
        model = model.to(device)

        Criterion = nn.CrossEntropyLoss(reduction='none')
        Criterion = Criterion.cuda()
        optimizer = torch.optim.Adam(model.parameters(), 0.001)

        # train task model
        trained_model = train(iteration,n_epochs,model,labeled_loader,val_loader,optimizer,Criterion)
        torch.save(trained_model, r"C:\Users\Rayeh\Desktop\Med_self-training_Yuan\ISIC2018_500\pretrain_resnet18_"+str(iteration)+".pth")
        
        # test task model
        # trained_model = torch.load(r"C:\Users\Rayeh\Desktop\Med_self-training_Yuan\ISIC2018_500\pretrain_resnet18_"+str(iteration)+".pth")
        t1,t2,t3,t4,testacc = test(test_loader, device,iteration)#看預測準確率多少
        logger.info(f'student model number : {iteration}, testing acc : {testacc}')
        alltest.append(testacc)
        big=testacc

        # predict pseudo label and feature
        logger.info(f'label data {len(labeled_indices)}: predict label and get feature')
        ext_feature_1, _, all_label_1, predict_softmax_1, _= test(labeled_loader, device,iteration)#對已label資料做特徵擷取
        logger.info(f'unlabel data {len(unlabeled_indices)}: predict label and get feature')
        ext_feature_2, _, all_label_2, predict_softmax_2, _= test(unlabeled_loader, device,iteration) #對unlabel資料做預測以及取特徵
        predict_softmax = predict_softmax_2
        ext_feature = np.concatenate([ext_feature_1,ext_feature_2])
        all_label = np.concatenate([all_label_1,all_label_2])

        # LP
        predict_softmax = np.array(predict_softmax)#轉成array
        ext_feature = np.array(ext_feature)#轉成array讓LB做處理
        all_label = np.array(all_label)
        lb_out, unlabel_idx = LB(ext_feature, all_label, labeled_indices)
        lb_out = lb_out[unlabel_idx]

        #  pseudo label prediction model
        feature = np.concatenate((lb_out[:len(predict_softmax)], predict_softmax),axis=1) #把兩個預測結果concat起來準備丟進去模型
        logger.info(f'concat feature: {np.shape(feature)}, all_label_2:{np.shape(all_label_2)}')
        feature_data = MyDataset(feature, all_label)
        fc_input_loader = DataLoader(dataset=feature_data,batch_size=64)
        modelfc = torch.load(r'C:\Users\Rayeh\Desktop\Med_self-training_Yuan\ISIC2018_500\FC20.pth')
        logger.info('use model and LB predict label to predict pesudo label...')
        pre_soft,pre_hard = fullyconnecttest(fc_input_loader,modelfc,device)
        
        #add pseudo-label data to labeled data
        labeled_indices ,unlabeled_indices,train_dataset, new_data= addpsudo(pre_soft,pre_hard,labeled_indices,unlabeled_indices,train_dataset,all_label_2)
        if new_data<=0:
            if patience==0:
                patience+=1
            else:
                logger.info(f'self-training finished!')
                logger.info(f'total iteration: {iteration}, final cycle acc: {testacc}, labeled: {len(labeled_indices)}, unlabeled: {len(unlabeled_indices)}')
                break
        else:
            patience=0
        labeled_sampler = SubsetRandomSampler(labeled_indices)
        unlabeled_sampler = SubsetRandomSampler(unlabeled_indices)
        logger.info(f'label: {Counter( np.array(train_dataset.target)[labeled_indices])}')
        logger.info(f'unlabel: {Counter( np.array(train_dataset.target)[unlabeled_indices])}')
        # labeled_loader = dataloader(labeled_dataset,batch_size) #finetune好的模型
        labeled_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=labeled_sampler)
        # unlabeled_loader = dataloader(unlabeled_dataset,batch_size)
        unlabeled_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=unlabeled_sampler)
        save_object(train_dataset, r'C:\Users\Rayeh\Desktop\Med_self-training_Yuan\ISIC2018_500\dataset_'+str(iteration))
        np.savez(r'C:\Users\Rayeh\Desktop\Med_self-training_Yuan\ISIC2018_500\dataset_indices_'+str(iteration)+'.npz',
                    labeled_indices=labeled_indices,
                    unlabeled_indices=unlabeled_indices)
        del modelfc

    plt.figure(figsize=(7,7),dpi=200)
    plt.plot(alltest,'b-o',label='test_curve')
    plt.title("test acc Curve")
    plt.xlabel("cycle")
    plt.xticks(list(range(1,iteration+1)))
    plt.ylabel("acc")
    plt.savefig(r'C:\Users\Rayeh\Desktop\Med_self-training_Yuan\picture\all_cycle_test_acc.png')
