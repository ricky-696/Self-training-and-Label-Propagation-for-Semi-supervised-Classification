import os
import torch
import logging
import random
import numpy as np
from dataset import Pseudo_data

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def get_logger(log_filename='Med_SelfTraining'):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(levelname)1.1s %(asctime)s %(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y%m%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = os.path.join('log', f'{log_filename}_{current_datetime}.log')

    log_dir = os.path.dirname(filename_with_timestamp)
    
    os.makedirs(log_dir, exist_ok=True)
        
    fh = logging.FileHandler(filename_with_timestamp)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


def get_original_dataset(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
        
    return dataset


def vis_pseudo_data_images(pseudo_data_loader, vis_batch_num, log_dir='.', log_filename='FC_result.tfevents'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename=f'{timestamp}_{log_filename}'
    
    # 創建 SummaryWriter
    writer = SummaryWriter(os.path.join(log_dir, log_filename))

    # 獲取指定批次數據
    for batch_idx, batch_data in enumerate(pseudo_data_loader):
        
        if batch_idx >= vis_batch_num:
            break

        # 解包數據
        batch = batch_data
        
        grouped_images = {}
        for i in range(batch['img'].shape[0]):
            label_value = batch['label'][i].item()
            
            if label_value not in grouped_images:
                grouped_images[label_value] = []
                
            grouped_images[label_value].append(batch['img'][i])

        for label_value, images in grouped_images.items():
            writer.add_images(f'pseudo_label: {label_value}', torch.stack(images, dim=0), global_step=batch_idx)

    # 關閉 SummaryWriter
    writer.close()


def vis_LP_pseudo_label(data_loader, sample_idx, LP_labels, batch_size, one_hot=True):
    pseudo_data_list = []
    
    dataset = get_original_dataset(data_loader.dataset)
        
    for i in range(len(LP_labels)):
        batch = dataset[sample_idx[i]]
        if one_hot:
            batch['label'] = np.argmax(LP_labels[i], axis=0)
        else:
            batch['label'] = LP_labels[i]

        pseudo_data_list.append(batch)

    pseudo_data = Pseudo_data(pseudo_data_list)
    vis_pseudo_data_images(
        torch.utils.data.DataLoader(pseudo_data, batch_size=batch_size),
        vis_batch_num=5, 
        log_dir=os.path.join('log', 'pseudo_labels'),
        log_filename='LP_result.tfevents'
    )

    return 0


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
        

if __name__ == '__main__':
    import subprocess
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