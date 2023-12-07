import os
import sys

sys.path.append('nnUNet')
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # GPU number
os.environ['nnUNet_raw'] = os.path.join(current_dir, 'nnunet_frame/DATASET/nnUnet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(current_dir, 'nnunet_frame/DATASET/nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(current_dir, 'nnunet_frame/DATASET/nnUNet_results')

from train_utils import *
from nnUNet.nnunetv2.run.run_training import get_trainer_from_args, run_training_entry
from nnUNet.nnunetv2.evaluation.find_best_configuration import find_best_configuration_entry_point
from nnUNet.nnunetv2.inference.predict_from_raw_data import predict_entry_point, nnUNetPredictor


if __name__ == "__main__":
    # nnUnet step: train -> find best config(if used full cross-validation) -> pred

    # nnunet_trainer = get_trainer_from_args(
    #     dataset_name_or_id='Dataset006_Lung', 
    #     configuration='2d',
    #     fold='all', 
    #     device=torch.device('cuda')
    # )

    # nnunet_trainer.run_training()
    # print('nnUNet model save at nnUNet_results')

    
    predict_entry_point()
    # Model's output at nnUNet/nnunetv2/inference/predict_from_raw_data.py: nnUNetPredictor -> _internal_maybe_mirror_and_predict()
    
    # predictor = nnUNetPredictor()
    
    # predictor.initialize_from_trained_model_folder(
    #     model_folder,
    #     args.f,
    #     checkpoint_name=args.chk
    # )
    
    # predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
    #                              overwrite=not args.continue_prediction,
    #                              num_processes_preprocessing=args.npp,
    #                              num_processes_segmentation_export=args.nps,
    #                              folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                              num_parts=args.num_parts,
    #                              part_id=args.part_id)