import os
import torch
import random
import pickle
import numpy as np

import subprocess
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter


def vis_pseudo_data_images(pseudo_data_loader, vis_batch_num, log_dir='.'):
    os.makedirs(log_dir, exist_ok=True)
    
    # 創建 SummaryWriter
    writer = SummaryWriter(log_dir)

    # 獲取指定批次數據
    for batch_idx, batch_data in enumerate(pseudo_data_loader):
        
        if batch_idx >= vis_batch_num:
            break

        # 解包數據
        img, label, omega = batch_data
        
        grouped_images = {}
        for i in range(img.shape[0]):
            label_value = label[i].item()
            
            if label_value not in grouped_images:
                grouped_images[label_value] = []
                
            grouped_images[label_value].append(img[i])

        for label_value, images in grouped_images.items():
            writer.add_images(f'pseudo_label: {label_value}', torch.stack(images, dim=0), global_step=batch_idx)

    # 關閉 SummaryWriter
    writer.close()


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


def select_balance_data(dataset, train_idx, n_test=70):
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
        

if __name__ == '__main__':
    from dataset import MNIST_omega
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset =  MNIST_omega(
        root='./mnist/',
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8
    )
    
    log_dir = 'log/vis_pseudo_data'
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir])
    
    vis_pseudo_data_images(
        dataloader, 
        vis_batch_num=5, 
        log_dir=log_dir
    )
    
    tensorboard_process.terminate()