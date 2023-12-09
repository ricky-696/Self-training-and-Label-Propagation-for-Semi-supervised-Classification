import torch
import trainer
from dataset import ISIC2018_Dataset
from opt import arg_parse


if __name__ == '__main__':
    args = arg_parse()
    args.num_classes = 7
    args.device = torch.device('cpu')
    model = torch.load('trained_model/ISIC2018/resnet18/best_student_cycle_5.pt').to(args.device)

    test_loader = torch.utils.data.DataLoader(
        dataset=ISIC2018_Dataset(type='test'),
        batch_size=1,
        shuffle=args.shuffle,
        num_workers=8
    )

    features, gt_label, pre_soft, data_idx, acc = trainer.predict(args, model, 'resnet', test_loader)
    print('ISIC testset acc:', acc)



    

