import os
import glob
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split

class MNIST_omega(Dataset):
    def __init__(self, root, train=True, transform=None, download=True, debug=False):
        super().__init__()
        if debug:
            self.mnist, self.classes = self.debug_mnist(root, train, transform, download)
        else:
            self.mnist = datasets.MNIST(root, train=train, transform=transform, download=download)
            self.classes = len(self.mnist.classes)

        self.omega = torch.tensor([1.] * len(self.mnist), dtype=torch.float32)

    def debug_mnist(self, root, train, transform, download):
        # Load MNIST dataset
        mnist = datasets.MNIST(root, train=train, transform=transform, download=download)

        # Get the number of classes
        num_classes = len(mnist.classes)

        # Initialize a list to keep track of whether each class has at least one sample
        class_samples = [False] * num_classes

        # Initialize a list to store the selected samples
        selected_samples = []

        # Iterate through the MNIST dataset
        for i in range(len(mnist)):
            img, target = mnist[i]

            # Check if the class has been encountered before
            if not class_samples[target]:
                # Add the sample to the list
                selected_samples.append((img, target))
                class_samples[target] = True

            # Check if all classes have at least one sample
            if all(class_samples):
                for j in range(i, i + num_classes):
                    img, target = mnist[j]
                    selected_samples.append((img, target))
                
                break

        # Return a new dataset with the selected samples
        return selected_samples, num_classes

    def __getitem__(self, index):
        img, target = self.mnist[index]

        # all need to be tensor
        batch = {}
        batch['img'] = img
        batch['label'] = torch.tensor(target)
        batch['omega'] = self.omega[index]
        batch['idx'] = torch.tensor(index)

        return batch

    def __len__(self):
        return len(self.mnist)
    
    def get_num_classes(self):
        return self.classes


class Concat_Pseudo_label_data(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x).float()
        self.label = y

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)
    

class Pseudo_data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # put new idx into Pseudo data
        self.data[index]['idx'] = torch.tensor(index)
        
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class ISIC_Dataset(Dataset):
    def __init__(self, type='train', transform=None):
        self.data = []
        self.target = []
        self.transform = transform
        self.omega = []
        if type=='train':
            data_glob = glob.glob(os.path.join(r'D:\Project\Med_AI\data\ISIC2018\ISIC2018_Task3_Training_Input','*.jpg'))
            GroundTruth = pd.read_csv(r'D:\Project\Med_AI\data\ISIC2018\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv',index_col='image')
        elif type=='valid':
            data_glob = glob.glob(os.path.join(r'D:\Project\Med_AI\data\ISIC2018\ISIC2018_Task3_Validation_Input','*.jpg'))
            GroundTruth = pd.read_csv(r'D:\Project\Med_AI\data\ISIC2018\ISIC2018_Task3_Validation_GroundTruth\ISIC2018_Task3_Validation_GroundTruth.csv',index_col='image')
        GroundTruth.columns = ['0','1','2','3','4','5','6']
        for i, data_path in enumerate(data_glob):
            self.data.append(data_path)
            self.omega.append(1.)
            data_name = data_path.split('\\')[-1].split('.')[0]
            target = int(GroundTruth.loc[data_name].index[GroundTruth.loc[data_name]==1].values)
            self.target.append(target)

    def __getitem__(self,index):
        data_path = self.data[index]
        label = self.target[index]
        omega = self.omega[index]
        data = Image.open(data_path).convert('RGB')
        if self.transform is not None:
            data = self.transform(data)
        return data, label, omega

    def __len__(self):
        return len(self.data)
    
    
if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    data_train = MNIST_omega(
            root='./mnist/',
            train=True,
            transform=transform,
            download=True,
            debug=False
    )
    
    train_data, unlabeled_data = random_split(data_train, [0.5, 0.5])
    
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=data_train,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=8
    # )
    
    for batch in data_train:
        print('here')
    