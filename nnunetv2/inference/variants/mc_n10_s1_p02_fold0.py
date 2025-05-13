from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data_monte_carlo_dropout import nnUNetPredictorMCDropout
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import numpy as np
import os
import shutil
import nibabel as nib

import datetime # for converting seconds to hours, minutes, seconds

 # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction




if __name__ == "__main__":
    dropout_p = '02' # prob in 10%
    dropout_s = '1'
    n_sim = 10
    cuda_device = 1
    n_cases = 20
    folds = (0,)

    # time estimation
    time_for_one_case_s = 70 # in seconds # depends on image size, tile_step_size
    time_s = n_cases * n_sim * len(folds) * time_for_one_case_s * 1.1 # 10% more time for overhead
    time = str(datetime.timedelta(seconds=time_s))
    print(f"Estimated time for {n_cases} cases with {n_sim} simulations and folds {folds}: {time}")



    ''' # numpy
    def categorical_entropy(p, eps=1e-8):
        """Compute entropy for multi-class probabilities."""
        p = np.clip(p, eps, 1.0)
        return -np.sum(p * np.log(p), axis=0)  # sum over class dimension
    '''

    # pytorch
    def categorical_entropy(p, eps=1e-8):
        """Compute entropy for multi-class probabilities."""
        p = p.clamp(min=eps)
        return -(p * p.log()).sum(dim=0)
    

    '''# numpy
    def compute_multiclass_uncertainty(mc_preds):
        """
        mc_preds: np.ndarray of shape (T, C, Z, Y, X) with softmax probabilities
        Returns:
            shannon_entropy: np.ndarray of shape (Z, Y, X)
            mutual_information: np.ndarray of shape (Z, Y, X)
        """
        # Mean over MC samples -> shape: (C, Z, Y, X)
        p_mean = np.mean(mc_preds, axis=0)
        
        # Predictive entropy: H(p_mean), shape: (Z, Y, X)
        shannon_entropy = categorical_entropy(p_mean)
        
        # Entropy per sample: H(p_t), shape: (T, Z, Y, X)
        entropies = np.array([categorical_entropy(mc_preds[t, :, :, :, :]) for t in range(mc_preds.shape[0])])
        
        # Expected entropy: mean over MC samples
        expected_entropy = np.mean(entropies, axis=0)
        
        # Mutual Information
        mutual_information = shannon_entropy - expected_entropy
        
        return shannon_entropy, expected_entropy, mutual_information
    '''
        
    # pytorch
    def compute_multiclass_uncertainty(mc_preds, eps=1e-8):
        """
        mc_preds: Tensor of shape (T, C, Z, Y, X)
        Returns:
            shannon_entropy: Tensor of shape (Z, Y, X)
            expected_entropy: Tensor of shape (Z, Y, X)
            mutual_info: Tensor of shape (Z, Y, X)
        """
        mean_pred = mc_preds.mean(dim=0)  # (C, Z, Y, X)
        shannon_entropy = categorical_entropy(mean_pred, eps)

        entropies = -(mc_preds.clamp(min=eps) * mc_preds.clamp(min=eps).log()).sum(dim=1)  # (T, Z, Y, X)
        expected_entropy = entropies.mean(dim=0)  # (Z, Y, X)

        mutual_info = shannon_entropy - expected_entropy
        return shannon_entropy, expected_entropy, mutual_info
    
    '''# numpy
    def compute_multiclass_variance(mc_preds, reduce='mean'):
        """
        mc_preds: np.ndarray of shape (T, C, Z, Y, X)
        reduce: 'mean' or 'max' to reduce per-class variance to voxel level
        Returns:
            voxel_variance: np.ndarray of shape (Z, Y, X)
        """
        # Compute variance across MC samples → shape: (C, Z, Y, X)
        var_across_mc = np.var(mc_preds, axis=0)  # variance over T

        if reduce == 'mean':
            voxel_variance = np.mean(var_across_mc, axis=0)  # average over classes
        elif reduce == 'max':
            voxel_variance = np.max(var_across_mc, axis=0)   # max over classes
        else:
            raise ValueError("reduce must be 'mean' or 'max'")
        
        return voxel_variance
    '''

    # pytorch
    def compute_multiclass_variance(mc_preds, reduce='mean'):
        """
        mc_preds: Tensor of shape (T, C, Z, Y, X)
        reduce: 'mean' or 'max' to aggregate class-wise variance
        Returns:
            voxel_variance: Tensor of shape (Z, Y, X)
        """
        var_across_mc = mc_preds.var(dim=0)  # (C, Z, Y, X)

        if reduce == 'mean':
            voxel_variance = var_across_mc.mean(dim=0)  # (Z, Y, X)
        elif reduce == 'max':
            voxel_variance = var_across_mc.max(dim=0).values  # (Z, Y, X)
        else:
            raise ValueError("reduce must be 'mean' or 'max'")

        return voxel_variance



    folder = join(nnUNet_raw, 'Dataset003_ImageCAS_split/imagesTs')

    cases = [os.path.splitext(f)[0].split('_')[1] for f in sorted(os.listdir(folder)) if os.path.isfile(join(folder, f))][:n_cases]

    # instantiate the nnUNetPredictorMCDropout
    predictor = nnUNetPredictorMCDropout(
        tile_step_size=0.5, # can change this for speedup, but then we need to turn off gaussian noise if we choose 1
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', cuda_device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    '''
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, f'Dataset003_ImageCAS_split/nnUNetTrainerDropout__p{dropout_p}_s{dropout_s}__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )'''

    print('Loading model...')
    predictor.initialize_from_trained_model_folder(
        join('/mnt/processing/oswald/nnUNet_results', f'Dataset003_ImageCAS_split/nnUNetTrainerDropout__p{dropout_p}_s{dropout_s}__3d_fullres'),
        use_folds=folds,
        checkpoint_name='checkpoint_best.pth',
    )


    for case in cases:
        indir = [join(folder, f'case_{case}_0000.nii.gz')]
        indirs = [indir for _ in range(n_sim)]

        #outdir = [join(nnUNet_raw, f'Dataset003_ImageCAS_split/imagesTs/case_{case}/case_{case}_sim_{n}.nii.gz')]
        temp_outdir = join(nnUNet_results, f'Dataset003_ImageCAS_split/temp/gpu_{cuda_device}_mc_n{n_sim}_p{dropout_p}_s{dropout_s}')
        temp_outdirs = [join(temp_outdir, f'sim_{i}.nii.gz') for i in range(n_sim)]
        os.makedirs(temp_outdir, exist_ok=True)

        print('making output directories')
        outdir = join(nnUNet_results, f'Dataset003_ImageCAS_split/mc_n{n_sim}_p{dropout_p}_s{dropout_s}/case_{case}')
        os.makedirs(outdir, exist_ok=True)
        # os.makedirs(temp_outdir, exist_ok=True)


        # run the prediction
        print(f"Predicting case {case} with {n_sim} simulations")
        predictor.predict_from_files(
            indirs,
            temp_outdirs,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0)
        

        
        # npz_files = [join(nnUNet_raw, f'Dataset003_ImageCAS_split/imagesTs_predfullres/case_{case}_sim_{n}.npz') for n in range(nsim)]
        npz_files = [join(temp_outdir, f'sim_{i}.nii.gz.npz') for i in range(n_sim)]

        # Load data from each file
        loaded_data = [np.load(f) for f in npz_files]

        # Assuming all files have the same keys
        keys = loaded_data[0].files  # e.g., ['arr_0']

        # Concatenate along a new dimension (e.g., axis=0)
        concatenated_data = {
            key: np.stack([data[key].astype(np.float16) for data in loaded_data], axis=0)
            for key in keys
        }

        # Save the concatenated result to a new file
        print(f"Saving concatenated data to {join(outdir, f'case_{case}_merged.npz')}")
        np.savez_compressed(join(outdir, f'case_{case}_merged.npz'), **concatenated_data)
        
        print(f"Deleting temporary files in {temp_outdir}")
        shutil.rmtree(temp_outdir)

        
        # calc var-uncertainty and mean

        data = np.load(join(outdir, f'case_{case}_merged.npz'))

        data = data[data.files[0]][:, :, :, :, :] # shape (mc, class, xyz)
        print(f"data shape: {data.shape}, dim: {data.ndim}")
        data = np.transpose(data, (0, 1, 4, 3, 2))
        print(f"data shape: {data.shape}, dim: {data.ndim}")

        data = torch.from_numpy(data).half().cuda(cuda_device)

        data_var = compute_multiclass_variance(data, reduce='mean')
        data_mean = data.mean(dim=0) # !!! this only saves the foreground class, can be changed to save all classes, but then wont work as viewable segmentation

        data_shannon_entropy, data_expected_entropy, data_mutual_info = compute_multiclass_uncertainty(data)

        '''
        print(f"data_var shape: {data_var.shape}, dim: {data_var.ndim}")
        print(f"data_mean shape: {data_mean.shape}, dim: {data_mean.ndim}")
        print(f"data_shannon_entropy shape: {data_shannon_entropy.shape}, dim: {data_shannon_entropy.ndim}")
        print(f"data_expected_entropy shape: {data_expected_entropy.shape}, dim: {data_expected_entropy.ndim}")
        print(f"data_mutual_info shape: {data_mutual_info.shape}, dim: {data_mutual_info.ndim}")
        '''

        # print(data_mean.sum(dim=0).max())

        affine = nib.load(indir[0]).affine

        # Save the data as NIfTI files
        data_var = data_var.cpu().numpy().astype(np.float32)
        data_mean = data_mean.cpu().numpy().astype(np.float32)
        data_shannon_entropy = data_shannon_entropy.cpu().numpy().astype(np.float32)
        data_expected_entropy = data_expected_entropy.cpu().numpy().astype(np.float32)
        data_mutual_info = data_mutual_info.cpu().numpy().astype(np.float32)

        nib.Nifti1Image(data_var, affine).to_filename(join(outdir, f'case_{case}_var.nii.gz'))
        nib.Nifti1Image(data_mean, affine).to_filename(join(outdir, f'case_{case}_mean.nii.gz'))
        nib.Nifti1Image(data_shannon_entropy, affine).to_filename(join(outdir, f'case_{case}_shannon_entropy.nii.gz'))
        nib.Nifti1Image(data_expected_entropy, affine).to_filename(join(outdir, f'case_{case}_expected_entropy.nii.gz'))
        nib.Nifti1Image(data_mutual_info, affine).to_filename(join(outdir, f'case_{case}_mutual_info.nii.gz'))
        print(f"Finished processing case {case}")



