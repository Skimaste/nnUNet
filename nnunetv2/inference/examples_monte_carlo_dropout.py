

if __name__ == '__main__':
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    import torch
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data_monte_carlo_dropout import nnUNetPredictorMCDropout
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction

    # instantiate the nnUNetPredictorMCDropout
    predictor = nnUNetPredictorMCDropout(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
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

    # print(predictor.network)
    # predict from raw data
    indir = join(nnUNet_raw, 'Dataset003_ImageCAS_split/imagesTs')
    outdir = join(nnUNet_raw, 'Dataset003_ImageCAS_split/imagesTs_predfullres_2')
    predictor.predict_from_files([[join(indir, 'case_0002_0000.nii.gz')],
                                [join(indir, 'case_0015_0000.nii.gz')]],
                                [join(outdir, 'case_0002.nii.gz'),
                                join(outdir, 'case_0015.nii.gz')],
                                save_probabilities=True, overwrite=True,
                                num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

   