from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data_monte_carlo_dropout import nnUNetPredictorMCDropout
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import numpy as np
import os
import shutil
import nibabel as nib

 # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction
if __name__ == "__main__":
    dropout_p = '02' # prob in 10%
    dropout_s = '2'
    n_sim = 3
    cuda_device = 2
    n_cases = 1

    def binary_entropy(p, eps=1e-8):
        p = np.clip(p, eps, 1 - eps)
        return -p * np.log(p) - (1 - p) * np.log(1 - p)

    def compute_uncertainty(mc_preds):
        """
        mc_preds: np.ndarray of shape (T, Z, Y, X) with probabilities
        Returns:
            predictive_entropy: np.ndarray of shape (Z, Y, X)
            mutual_information: np.ndarray of shape (Z, Y, X)
        """
        # Predictive mean
        p_mean = np.mean(mc_preds, axis=0)  # shape: (Z, Y, X)
        
        # Entropy of the predictive mean (total uncertainty)
        predictive_entropy = binary_entropy(p_mean)
        
        # Entropy of each MC sample
        entropies = binary_entropy(mc_preds)  # shape: (T, Z, Y, X)
        
        # Expected entropy (aleatoric uncertainty)
        expected_entropy = np.mean(entropies, axis=0)  # shape: (Z, Y, X)
        
        # Mutual Information (epistemic uncertainty)
        mutual_information = predictive_entropy - expected_entropy
        
        return predictive_entropy, mutual_information

    folder = join(nnUNet_raw, 'Dataset003_ImageCAS_split/imagesTs')

    cases = [os.path.splitext(f)[0].split('_')[1] for f in sorted(os.listdir(folder)) if os.path.isfile(join(folder, f))][:n_cases]

    # instantiate the nnUNetPredictorMCDropout
    predictor = nnUNetPredictorMCDropout(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', cuda_device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, f'Dataset003_ImageCAS_split/nnUNetTrainerDropout__p{dropout_p}_s{dropout_s}__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )


    for case in cases:
        indir = [join(folder, f'case_{case}_0000.nii.gz')]
        indirs = [indir for _ in range(n_sim)]

        #outdir = [join(nnUNet_raw, f'Dataset003_ImageCAS_split/imagesTs/case_{case}/case_{case}_sim_{n}.nii.gz')]
        temp_outdir = join(nnUNet_results, f'Dataset003_ImageCAS_split/temp')
        temp_outdirs = [join(temp_outdir, f'sim_{i}.nii.gz') for i in range(n_sim)]

        outdir = join(nnUNet_results, f'Dataset003_ImageCAS_split/case_{case}')

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
        npz_files = [join(temp_outdir, f'sim_{i}.npz') for i in range(n_sim)]

        # Load data from each file
        loaded_data = [np.load(f) for f in npz_files]

        # Assuming all files have the same keys
        keys = loaded_data[0].files  # e.g., ['arr_0']

        # Concatenate along a new dimension (e.g., axis=0)
        concatenated_data = {
            key: np.stack([data[key].astype(np.float32) for data in loaded_data], axis=0)
            for key in keys
        }

        # Save the concatenated result to a new file
        np.savez_compressed(join(outdir, f'case_{case}_merged.npz'), **concatenated_data)
        
        shutil.rmtree(temp_outdir)

        # calc var-uncertainty and mean

        data = np.load(join(outdir, f'case_{case}_merged.npz'))

        data = data[data.files[0]][:, 1, :, :, :]

        data_var = np.var(data, axis = 0)
        data_mean = np.mean(data, axis = 0)

        data_pred_entropy, data_mutual_info = compute_uncertainty(data)

        affine = nib.load(indir[0]).affine

        nib.Nifti1Image(data_var, affine).to_filename(join(outdir, f'case_{case}_var.nii.gz'))
        nib.Nifti1Image(data_mean, affine).to_filename(join(outdir, f'case_{case}_mean.nii.gz'))
        nib.Nifti1Image(data_pred_entropy, affine).to_filename(join(outdir, f'case_{case}_pred_entropy.nii.gz'))
        nib.Nifti1Image(data_mutual_info, affine).to_filename(join(outdir, f'case_{case}_mutual_info.nii.gz'))




