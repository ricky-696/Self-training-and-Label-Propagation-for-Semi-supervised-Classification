import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False, help='open/close debug mode')
    parser.add_argument('--devices', nargs='+', type=str, default='0', help='gpu number')
    
    # train
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--teacher_epochs', type=int, default=2, help='epochs of finetune pretrain model')
    parser.add_argument('--student_epochs', type=int, default=10, help='epochs of training student')
    parser.add_argument('--max_self_training_iteration', type=int, default=10, help='max_self_training_iteration')
    
    parser.add_argument('--img_size', type=tuple, default=(256, 256), help='input img_size')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = arg_parse()
    print(args.debug)