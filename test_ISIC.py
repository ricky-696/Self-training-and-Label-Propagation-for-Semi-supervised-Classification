import os
import trainer
import numpy as np

import torch
import torch.nn.functional as F

from opt import arg_parse
from utils import get_logger
from dataset import ISIC2018_SRC
from torchvision import transforms
from SRC_MT.code.utils.metrics import compute_metrics_test


if __name__ == '__main__':
    args = arg_parse()
    args.num_classes = 7
    # args.device = torch.device('cpu')
    args.device = torch.device('cuda:1')

    args.model_type = 'densenet'
    args.title = 'ISIC2018_epoch60_20%_affine_dense'
    logger = get_logger(f'Test_{args.title}')

    args.test_model_dir = os.path.join('trained_model', args.title, args.model_type, 'student_best.pt')
    model = torch.load(args.test_model_dir).to(args.device)


    args.data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    test_loader = torch.utils.data.DataLoader(
        dataset=ISIC2018_SRC(type='testing', transform=args.data_transforms['test']),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=8
    )

    features, gt_label, pre_soft, data_idx, acc = trainer.predict(args, model, args.model_type, test_loader)
    logger.info(f'ISIC testset acc: {acc}')

    gt_tensor = torch.tensor(gt_label)
    gt_one_hot = F.one_hot(gt_tensor)

    pre_soft = torch.tensor(pre_soft)

    AUROCs, Accus, Senss, Specs, Pre, F1 = compute_metrics_test(gt=gt_one_hot, pred=pre_soft)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    pre_avg = np.array(Pre).mean()
    F1_avg = np.array(F1).mean()

    logger.info(
        "TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST pre: {:6f}, TEST F1: {:6f}".format(
            AUROC_avg, 
            Accus_avg, 
            Senss_avg, 
            Specs_avg, 
            pre_avg, 
            F1_avg
        )
    )

