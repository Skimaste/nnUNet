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


class MonteCarloInference:
    def __init__(self,
                 dataset_name,
                 model,
                 n_sim,
                 folds,
                 n_cases,
                 cuda_device,
                 entropy = True,
                 variance = True,
                 mean = True
                 ):
        self.dataset_name = dataset_name
        self.indir = join(nnUNet_raw, dataset_name, 'imagesTs')
        self.model_path = join(nnUNet_results, dataset_name, model)
        self.outdir = join(self.model_path, 'inference')
        self.n_sim = n_sim
        self.folds = folds
        self.n_cases = n_cases
        self.cuda_device = cuda_device
        self.entropy = entropy
        self.variance = variance
        self.mean = mean

        time_for_one_case_s = 70 # in seconds # depends on image size, tile_step_size
        time_s = n_cases * n_sim * len(folds) * time_for_one_case_s * 1.1 # 10% more time for overhead
        time = str(datetime.timedelta(seconds=time_s))

        self.time_estimation = time
        print(f"Estimated time for {n_cases} cases with {n_sim} simulations and folds {folds}: {time}")


    def categorical_entropy(self, p, eps=1e-8):
        """Compute entropy for multi-class probabilities."""
        p = p.clamp(min=eps)
        return -(p * p.log()).sum(dim=0)
    

    def compute_multiclass_uncertainty(self, mc_preds, eps=1e-8):
        """
        mc_preds: Tensor of shape (T, C, Z, Y, X)
        Returns:
            shannon_entropy: Tensor of shape (Z, Y, X)
            expected_entropy: Tensor of shape (Z, Y, X)
            mutual_info: Tensor of shape (Z, Y, X)
        """
        mean_pred = mc_preds.mean(dim=0)  # (C, Z, Y, X)
        shannon_entropy = self.categorical_entropy(mean_pred, eps)

        entropies = -(mc_preds.clamp(min=eps) * mc_preds.clamp(min=eps).log()).sum(dim=1)  # (T, Z, Y, X)
        expected_entropy = entropies.mean(dim=0)  # (Z, Y, X)

        mutual_info = shannon_entropy - expected_entropy
        return shannon_entropy, expected_entropy, mutual_info


    def compute_multiclass_variance(self, mc_preds, reduce='mean'):
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
    
    def compute_multiclass_mean(self, mc_preds):
        """
        mc_preds: Tensor of shape (T, C, Z, Y, X)
        Returns:
            voxel_mean: Tensor of shape (C, Z, Y, X)
        """
        mean_across_mc = mc_preds.mean(dim=0)
        return mean_across_mc
    
    def run_inference(self):
        predictor = nnUNetPredictorMCDropout(
            tile_step_size=0.5, # can change this for speedup, but then we need to turn off gaussian noise if we choose 1
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', self.cuda_device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        predictor.initialize_from_trained_model_folder(
            join('/mnt/processing/oswald/nnUNet_results', f'Dataset003_ImageCAS_split/nnUNetTrainerDropout__p05_s2__3d_fullres'), # self.model_path
            use_folds=self.folds,
            checkpoint_name='checkpoint_final.pth',
        )

        cases = [os.path.splitext(f)[0].split('_')[1] for f in sorted(os.listdir(self.indir)) if os.path.isfile(join(self.indir, f))][:self.n_cases]

        for case in cases: 
            folder = [join(self.indir, f'case_{case}_0000.nii.gz')]
            folder_sim = [folder for _ in range(self.n_sim)]

            temp_outdir = join(self.outdir, f'temp/gpu_{self.cuda_device}')
            os.makedirs(temp_outdir, exist_ok=True)
            temp_outdirs = [join(temp_outdir, f'sim_{i}.nii.gz') for i in range(self.n_sim)]

            predictor.predict_from_files(
                folder_sim,
                temp_outdirs,
                save_probabilities=True,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0)

            npz_files = [join(temp_outdir, f'sim_{i}.nii.gz.npz') for i in range(self.n_sim)]

            loaded_data = [np.load(f) for f in npz_files]

            keys = loaded_data[0].files  # e.g., ['arr_0']

            concatenated_data = {
                key: np.stack([data[key].astype(np.float16) for data in loaded_data], axis=0)
                for key in keys
            }

            np.savez_compressed(join(self.outdir, f'{case}/case_{case}_merged.npz'), **concatenated_data)

            shutil.rmtree(temp_outdir)

    def compute_uncertainty(self):

        cases = [os.path.splitext(f)[0].split('_')[1] for f in sorted(os.listdir(self.indir)) if os.path.isfile(join(self.indir, f))][:self.n_cases]

        for case in cases:
            data = np.load(join(self.outdir, f'{case}/case_{case}_merged.npz'))

            data = data[data.files[0]][:, :, :, :, :]

            data = np.transpose(data, (0, 1, 4, 3, 2))

            data = torch.from_numpy(data).half().cuda(self.cuda_device)

            # Compute uncertainty metrics
            if self.variance:
                voxel_variance = self.compute_multiclass_variance(data, reduce='mean')
                self.save_image(voxel_variance, self.outdir, case, 'variance')
            if self.mean:
                voxel_mean = self.compute_multiclass_mean(data)
                self.save_image(voxel_mean, self.outdir, case, 'mean')
            if self.entropy:
                shannon_entropy, expected_entropy, mutual_info = self.compute_multiclass_uncertainty(data)
                self.save_image(shannon_entropy, self.outdir, case, 'shannon_entropy')
                self.save_image(expected_entropy, self.outdir, case, 'expected_entropy')
                self.save_image(mutual_info, self.outdir, case, 'mutual_info')


    def save_image(self, data, outdir, case, metric):
        # Save the uncertainty metric as a NIfTI file
        data = data.cpu().numpy().astype(np.float32)
        # data = np.transpose(data, (2, 3, 4, 0, 1))
        # data = np.squeeze(data)

        affine = nib.load(join(self.indir, f'case_{case}_0000.nii.gz')).affine

        img = nib.Nifti1Image(data, affine=affine)
        nib.save(img, join(outdir, f'{case}/case_{case}_{metric}.nii.gz'))

    def run(self):
        self.run_inference()
        self.compute_uncertainty()

if __name__ == "__main__":
    mc_inference = MonteCarloInference(
        dataset_name = 'Dataset003_ImageCAS_split',
        model='nnUNetTrainerDropout__p05_s2__3d_fullres',
        n_cases=2,
        n_sim=3,
        folds=(0,),
        cuda_device=0,
        variance=True,
        mean=True,
        entropy=True
    )

    # mc_inference.run()
    mc_inference.compute_uncertainty()