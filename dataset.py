import os
import glob
import torch
import pandas as pd
import nibabel as nib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split

"""
If You Want Custom Datset, __getitem__() need return a dict 'batch':
batch include following data: (All dtype need to be tensor)
    batch['img'] = Your image (dtype: torch.float32)
    batch['label'] = Your label (dtype: torch.int64)
    batch['omega'] = class weights (dtype: torch.float32)
    batch['idx'] = torch.tensor(dtype: torch.int64)
"""

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


class ISIC2018_Dataset(Dataset):
    def __init__(self, type='train', data_dir=os.path.join('Datasets', 'ISIC2018'), transform=transforms.ToTensor()):
        self.data = []
        self.target = []
        self.transform = transform

        if type=='train':
            data_glob = glob.glob(os.path.join(data_dir, 'ISIC2018_Task3_Training_Input', '*.jpg'))
            GroundTruth = pd.read_csv(os.path.join(data_dir, 'ISIC2018_Task3_Training_GroundTruth', 'ISIC2018_Task3_Training_GroundTruth.csv'), index_col='image')
        elif type=='valid':
            data_glob = glob.glob(os.path.join(data_dir, 'ISIC2018_Task3_Validation_Input', '*.jpg'))
            GroundTruth = pd.read_csv(os.path.join(data_dir, 'ISIC2018_Task3_Validation_GroundTruth', 'ISIC2018_Task3_Validation_GroundTruth.csv'), index_col='image')
        elif type=='test':
            data_glob = glob.glob(os.path.join(data_dir, 'ISIC2018_Task3_Test_Input', '*.jpg'))
            GroundTruth = pd.read_csv(os.path.join(data_dir, 'ISIC2018_Task3_Test_GroundTruth', 'ISIC2018_Task3_Test_GroundTruth.csv'), index_col='image')

        GroundTruth.columns = ['0','1','2','3','4','5','6']
        self.omega = torch.tensor([1.] * len(data_glob), dtype=torch.float32)

        for data_path in data_glob:
            self.data.append(data_path)
            
            data_name = data_path.split('/')[-1].split('.')[0] # if OS not Linux, need change('/')
            target = int(GroundTruth.loc[data_name][GroundTruth.loc[data_name]==1].index[0])
            self.target.append(target)

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        batch = {}
        batch['img'] = img
        batch['label'] = torch.tensor(self.target[index])
        batch['omega'] = self.omega[index]
        batch['idx'] = torch.tensor(index)

        return batch

    def __len__(self):
        return len(self.data)
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_data = ISIC2018_Dataset(type='train', transform=transform)
    
    num_of_cls = {}
    for batch in tqdm(train_data):
        if not int(batch['label']) in num_of_cls.keys():
            num_of_cls[int(batch['label'])] = 1
        else:
            num_of_cls[int(batch['label'])] += 1
    
    labels = list(num_of_cls.keys())
    frequencies = list(num_of_cls.values())

    # Plotting the histogram
    plt.bar(labels, frequencies, color='blue', alpha=0.7)
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.title('Class Distribution in Training Data')
    plt.savefig('cls.png')
