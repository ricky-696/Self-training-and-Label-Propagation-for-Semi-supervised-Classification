from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from tqdm import tqdm

if __name__ == "__main__":
    # device = 1
    
    folder = '/home/ltc110u/nnUNet/nnunet_frame/DATASET/nnUnet_preprocessed/Dataset006_Lung/nnUNetPlans_3d_lowres'
    
    # num_images_properties_loading_threshold 最多可以讀的資料筆數
    ds = nnUNetDataset(folder , num_images_properties_loading_threshold=1000)
    print('ok')
    ''' def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False):
    '''
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    # no label manager
    
    
    # lung_data_loader = nnUNetDataLoaderBase(
    #     data_dir = '',
    #     data_augumention = []
    # )
    
    # model = nn_unet(
    #     layer=4,
    #     normalized=True
    # )
    
    # tbar = tqdm(lung_data_loader)
    
    # for img, label in tbar:
        
    #     img, label = img.to(device), label.to(device)
    #     out = model(img)
        
    #     print(out.shape)
        
        
    
    
    