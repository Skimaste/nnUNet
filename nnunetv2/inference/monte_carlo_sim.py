if __name__ == '__main__':
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    import torch
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data_monte_carlo_dropout import nnUNetPredictorMCDropout
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import numpy as np
    import os

    # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction

    # instantiate the nnUNetPredictorMCDropout
    predictor = nnUNetPredictorMCDropout(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 2),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s2__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    
    cases = ['0002', '0015']
    nsim = 5

    for case in cases:
        indir = [[join(nnUNet_raw, f'Dataset003_ImageCAS_split/imagesTs/case_{case}_0000.nii.gz')]]
        '''
        for n in range(nsim):
            outdir = [join(nnUNet_raw, f'Dataset003_ImageCAS_split/imagesTs/case_{case}/case_{case}_sim_{n}.nii.gz')]
            output = predictor.predict_from_files(
                indir,
                outdir,
                save_probabilities=True,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0)
        '''

        npz_files = [join(nnUNet_raw, f'Dataset003_ImageCAS_split/imagesTs_predfullres/case_{case}_sim_{n}.npz') for n in range(nsim)]

        # Load data from each file
        loaded_data = [np.load(f) for f in npz_files]

        # Assuming all files have the same keys
        keys = loaded_data[0].files  # e.g., ['arr_0']

        # Concatenate along a new dimension (e.g., axis=0)
        concatenated_data = {
            key: np.stack([data[key] for data in loaded_data], axis=0)
            for key in keys
        }

        # Save the concatenated result to a new file
        np.savez(join(nnUNet_raw, f'Dataset003_ImageCAS_split/imagesTs_predfullres/case_{case}_merged.npz'), **concatenated_data)

        # Optional: remove original files
        #for f in npz_files:
            #os.remove(f)










