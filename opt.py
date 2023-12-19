import os
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False, help='open/close debug mode')
    parser.add_argument('--devices', type=str, default='1', help='gpu number')
    
    # dir
    parser.add_argument('--dataset_dir', type=str, default='Datasets/mnist', help='dataset_dir')
    parser.add_argument('--save_model_dir', type=str, default=os.path.join('trained_model', 'MNIST', 'resnet18'), help='save_model_dir')
    parser.add_argument('--test_model_dir', type=str, default='trained_model/ISIC2018/resnet18/best.pt', help='test_model_dir')
    parser.add_argument('--log_filename', type=str, default='MNIST_SelfTraining', help='log_filename')
    
    # dataloader & Visualization
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset')
    parser.add_argument('--vis_pseudo', type=bool, default=False, help='If your want to visualize your pseudo label, set True')

    # train
    parser.add_argument('--binary_cls', type=bool, default=False, help='If your task is binary classification, set True')
    parser.add_argument('--pretrain', type=bool, default=True, help='If your model need pretrain, set True')
    parser.add_argument('--pueudo_label_pred_model', type=str, default='avg', help='pueudo_label_pred_model type: FC or avg')
    parser.add_argument('--model_type', type=str, default='resnet', help='backbone model_type')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--teacher_epochs', type=int, default=2, help='epochs of finetune pretrain model')
    parser.add_argument('--student_epochs', type=int, default=5, help='epochs of training student')
    parser.add_argument('--FC_epochs', type=int, default=200, help='epochs of training student')
    parser.add_argument('--max_self_training_iteration', type=int, default=10, help='max_self_training_iteration')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='input img_size')
    
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = arg_parse()
    print(args.debug)