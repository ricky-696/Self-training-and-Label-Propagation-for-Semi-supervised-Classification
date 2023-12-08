import os
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False, help='open/close debug mode')
    parser.add_argument('--devices', type=str, default='1', help='gpu number')
    
    # dir
    parser.add_argument('--dataset_dir', type=str, default='Datasets/mnist', help='dataset_dir')
    parser.add_argument('--save_model_dir', type=str, default=os.path.join('trained_model', 'MNIST', 'resnet18'), help='save_model_dir')
    parser.add_argument('--log_filename', type=str, default='log/MNIST_SelfTraining.log', help='log_filename')
    
    # dataloader
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset')
    
    # train
    parser.add_argument('--pretrain', type=bool, default=True, help='If your model need pretrain, set True')
    parser.add_argument('--model_type', type=str, default='resnet', help='backbone model_type')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--teacher_epochs', type=int, default=1, help='epochs of finetune pretrain model')
    parser.add_argument('--student_epochs', type=int, default=5, help='epochs of training student')
    parser.add_argument('--FC_epochs', type=int, default=1000, help='epochs of training student')
    parser.add_argument('--max_self_training_iteration', type=int, default=10, help='max_self_training_iteration')
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='input img_size')
    
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = arg_parse()
    print(args.debug)