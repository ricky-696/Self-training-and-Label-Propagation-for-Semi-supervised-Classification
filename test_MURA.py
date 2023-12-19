import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score

from utils import get_logger
from DenseNet_MURA_PyTorch.pipeline import get_study_level_data, get_dataloaders


def pred_MURA(model, dataloader, logger):
    model.eval()
    
    preds, gts = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # get all data for one patient
            img = batch['images'][0].to(device)
            pred = model(img)
            pred = torch.argmax(pred, dim=1)
            
            # vote result
            counts = torch.bincount(pred)
            voted = torch.argmax(counts)
            
            preds.append(int(voted.cpu()))
            gts.append(int(batch['label'][0].cpu()))
        
        gts, preds = np.array(gts), np.array(preds)
        kappa_score = cohen_kappa_score(gts, preds)
        accuracy = accuracy_score(gts, preds)

        logger.info(f'Kappa Score: {kappa_score}')
        logger.info(f'Accuracy: {accuracy * 100:.2f}%')
        
    return kappa_score
            

if __name__ == '__main__':
    BATCH_SIZE = 1
    assert BATCH_SIZE == 1, "batch_size need to be 1, can't be more"

    data_cat = ['valid']
    model_type = 'resnet'
    data_dir = os.path.join('Datasets', 'MURA-v1.1')
    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    study_types = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
    logger = get_logger('Test_MURA')

    kappa_scores = []
    for study_type in study_types:
        model = torch.load(os.path.join('trained_model', 'MURA-v1.1', study_type, model_type, 'student_best.pt')).to(device)
        study_data = get_study_level_data(data_dir, study_type=study_type, data_cat=data_cat)
        dataloaders = get_dataloaders(study_data, batch_size=BATCH_SIZE, data_cat=data_cat)
        
        logger.info(f'Start predict {study_type}')
        kappa_scores.append(pred_MURA(model, dataloaders['valid'], logger))
        
    avg_kappa = np.mean(kappa_scores)
    logger.info(f'Avg Kappa Score: {avg_kappa}')
        
    

