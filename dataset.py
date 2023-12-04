import os
import glob
import torch
import pandas as pd
import nibabel as nib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

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
        return (img, torch.tensor(target), self.omega[index])

    def __len__(self):
        return len(self.mnist)
    
    def get_num_classes(self):
        return self.classes


class Concat_Psuedo_label_data(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x).float()
        self.label = y

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)
    

class Psuedo_data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self,index):
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
<<<<<<< HEAD
    
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
                
        
=======
    
>>>>>>> a24d2593845665f540f75225484a1191d26a8945
