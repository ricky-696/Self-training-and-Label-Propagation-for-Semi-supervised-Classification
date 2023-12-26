import torch
import trainer
from torchvision import transforms
from dataset import ISIC2018_Dataset
from opt import arg_parse


if __name__ == '__main__':
    args = arg_parse()
    args.num_classes = 7
    # args.device = torch.device('cpu')
    args.device = torch.device('cuda:0')
    args.batch_size = 16
    args.test_model_dir = 'trained_model/ISIC2018_epoch60_all_affine_res/resnet/pretrain/best.pt'
    model = torch.load(args.test_model_dir).to(args.device)


    args.data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    test_loader = torch.utils.data.DataLoader(
        dataset=ISIC2018_Dataset(type='test', transform=args.data_transforms['test']),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=8
    )

    features, gt_label, pre_soft, data_idx, acc = trainer.predict(args, model, 'resnet', test_loader)
    print('ISIC testset acc:', acc)
