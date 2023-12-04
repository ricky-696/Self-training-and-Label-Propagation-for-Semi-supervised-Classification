import os
import glob
import torch
import pandas as pd
import nibabel as nib

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import datasets, transforms

class MNIST_omega(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__()
        self.mnist = datasets.MNIST(root, train=train, transform=transform, download=True)
        self.train = train
        self.omega = [1.] * len(self.mnist)

    def __getitem__(self, index):
        img, target = self.mnist[index]

        return img, target, self.omega[index]

    def __len__(self):
        return len(self.mnist)
    
    def get_num_classes(self):
        return len(self.mnist.classes)


class MyDataset(Dataset):
    def __init__(self,x,y):
        self.data = torch.from_numpy(x)
        self.label = torch.from_numpy(y)

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)
    

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


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
    
# class MSD_Dataset(Dataset):
#     def __init__(self, type = 'train', transform = None):
#         self.data = []
#         self.transform = transform
#         if type == 'train':
#             data_glob = glob.glob('/home/ltc110u/Task06_Lung/imagesTr/*.nii.gz')
#             label_glob = glob.glob('/home/ltc110u/Task06_Lung/labelsTr/*.nii.gz')
#         elif type == 'valid':
#             data_glob = glob.glob('/home/ltc110u/Task06_Lung/imagesTs/*.nii.gz')
#         for i, data_path in enumerate(data_glob):
#             self.data.append(data_path)
#             data_name = data_path.split('/')[-1].split('.')[0]
            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         data_path = self.data[index]
#         image = nib.load(data_path)
                
        