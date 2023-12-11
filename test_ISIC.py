import torch
import trainer
from dataset import ISIC2018_Dataset
from opt import arg_parse


if __name__ == '__main__':
    args = arg_parse()
    args.num_classes = 7
    # args.device = torch.device('cpu')
    args.device = torch.device('cuda:1')
    args.batch_size = 16
    args.test_model_dir = 'trained_model/ISIC2018/resnet18/best_student_cycle_5.pt'
    model = torch.load(args.test_model_dir).to(args.device)

    test_loader = torch.utils.data.DataLoader(
        dataset=ISIC2018_Dataset(type='test'),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=8
    )

    features, gt_label, pre_soft, data_idx, acc = trainer.predict(args, model, 'resnet', test_loader)
    print('ISIC testset acc:', acc)



    

