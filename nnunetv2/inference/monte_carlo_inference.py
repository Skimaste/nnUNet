from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join, listdir
from nnunetv2.inference.predict_from_raw_data_monte_carlo_dropout import nnUNetPredictorMCDropout
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import numpy as np
import os
import shutil
import nibabel as nib
import re

import datetime # for converting seconds to hours, minutes, seconds


class MonteCarloInference:
    def __init__(self,
                 dataset_name,
                 model,
                 n_sim,
                 folds,
                 n_cases,
                 cuda_device,
                 config_name=None,
                 tta = True,
                 entropy = True,
                 variance = True,
                 mean = True
                 ):
        self.config_name = config_name
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
        self.model = model
        self.tta = tta
        if self.config_name:
            self.outdir = join(self.model_path, 'inference', self.config_name)
        

    def calc_time(self, print_time=True):
        time_for_one_case_s = 70 # in seconds # depends on image size, tile_step_size
        time_s = self.n_cases * self.n_sim * len(self.folds) * time_for_one_case_s * 1.1 # 10% more time for overhead
        time = str(datetime.timedelta(seconds=time_s))

        self.time_estimation = time
        
        if print_time:
            print(f"Estimated time for {self.n_cases} cases with {self.n_sim} simulations and folds {self.folds}: {self.time_estimation}")
        return self.time_estimation


    def categorical_entropy(self, p, dim, eps):
        """Compute entropy for multi-class probabilities."""
          # Log2 as we have two classes
        entropy = -(p * p.log2().nan_to_num()).sum(dim=dim).nan_to_num()
        return entropy / torch.log2(torch.tensor(p.size(dim)))


    def compute_multiclass_uncertainty(self, mc_preds, eps=1e-6):
        """
        mc_preds: Tensor of shape (T, C, Z, Y, X)
        Returns:
            shannon_entropy: Tensor of shape (Z, Y, X)
            expected_entropy: Tensor of shape (Z, Y, X)
            mutual_info: Tensor of shape (Z, Y, X)
        """

        mean_pred = mc_preds.mean(dim=0)  # (C, Z, Y, X)
        shannon_entropy = self.categorical_entropy(mean_pred, dim=0, eps=eps)  # (Z, Y, X)

        # entropies = -(mc_preds.clamp(min=eps) * mc_preds.clamp(min=eps).log()).sum(dim=1)  # (T, Z, Y, X) # gives nans
        entropies = self.categorical_entropy(mc_preds, dim=1, eps=eps)  # (T, Z, Y, X)
        expected_entropy = entropies.mean(dim=0)  # (Z, Y, X)

        mutual_info = torch.relu(shannon_entropy - expected_entropy)
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
            use_mirroring=self.tta,
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

            os.makedirs(join(self.outdir, f'case_{case}'), exist_ok=True)

            np.savez_compressed(join(self.outdir, f'case_{case}/case_{case}_merged.npz'), **concatenated_data)

            shutil.rmtree(temp_outdir)

    def compute_uncertainty(self):

        cases = [os.path.splitext(f)[0].split('_')[1] for f in sorted(os.listdir(self.indir)) if os.path.isfile(join(self.indir, f))][:self.n_cases]

        for case in cases:
            data = np.load(join(self.outdir, f'case_{case}/case_{case}_merged.npz'))

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

        os.makedirs(join(outdir, f'case_{case}'), exist_ok=True)

        affine = nib.load(join(self.indir, f'case_{case}_0000.nii.gz')).affine

        img = nib.Nifti1Image(data, affine=affine)
        nib.save(img, join(outdir, f'case_{case}/case_{case}_{metric}.nii.gz'))

    def save_image_clean(self, data, outdir, case, metric):
        # Save the uncertainty metric as a NIfTI file
        os.makedirs(join(outdir, f'{case}'), exist_ok=True)

        affine = nib.load(join(self.indir, f'{case}_0000.nii.gz')).affine

        img = nib.Nifti1Image(data, affine=affine)
        nib.save(img, join(outdir, f'{case}/{case}_{metric}.nii.gz'))

    def run(self):
        self.calc_time()
        self.run_inference()
        self.compute_uncertainty()


    def find_cases(self):
        cases = listdir(self.model_path)
        cases = [case for case in cases if re.match(r'case_\d+', case)]
        return cases

    def give_seg_and_foreground(self):
        cases = self.find_cases()
        for case in cases:
            mean_file_path = join(self.model_path, case, f'{case}_mean.nii.gz')

            mean_image = nib.load(mean_file_path).get_fdata()

            seg_image = np.argmax(mean_image, axis=0)
            seg_image = seg_image.astype(np.uint8)

            foreground_image = mean_image[1, :, :, :]
            foreground_image = foreground_image.astype(np.float32)

            self.save_image_clean(seg_image, self.model_path, case, 'seg')
            self.save_image_clean(foreground_image, self.model_path, case, 'foreground')




if __name__ == "__main__":

    nnUNet_results = '/mnt/processing/emil/nnUNet_results'
    
    models = ['base', 'tta', 'ens', 'tta_ens']

    for mod in models:
        mc_inference = MonteCarloInference(
            dataset_name = 'Dataset003_ImageCAS_split',
            model=join('nnUNetTrainerDropout__p00_s2__3d_fullres', mod),
            n_cases=20,
            n_sim=30,
            folds=(0,),
            cuda_device=3,
            config_name=None,
            tta=True,
            variance=True,
            mean=True,
            entropy=True
        )
        
        # mc_inference.run()
        # mc_inference.compute_uncertainty()
        mc_inference.give_seg_and_foreground()

    models = ['mcd', 'tta_mcd', 'ens_mcd', 'all']

    for mod in models:
        mc_inference = MonteCarloInference(
            dataset_name = 'Dataset003_ImageCAS_split',
            model=join('nnUNetTrainerDropout__p02_s2__3d_fullres', mod),
            n_cases=20,
            n_sim=30,
            folds=(0,),
            cuda_device=3,
            config_name=None,
            tta=True,
            variance=True,
            mean=True,
            entropy=True
        )
        
        # mc_inference.run()
        # mc_inference.compute_uncertainty()
        mc_inference.give_seg_and_foreground()
    
    '''
    mc_inference = MonteCarloInference(
        dataset_name = 'Dataset003_ImageCAS_split',
        model='nnUNetTrainerDropout__p02_s2__3d_fullres',
        n_cases=20,
        n_sim=30,
        folds=(0,),
        cuda_device=3,
        tta=True,
        variance=True,
        mean=True,
        entropy=True
    )

    # mc_inference.run()
    mc_inference.compute_uncertainty()

    mc_inference = MonteCarloInference(
        dataset_name = 'Dataset003_ImageCAS_split',
        model='nnUNetTrainerDropout__p01_s2__3d_fullres',
        n_cases=20,
        n_sim=30,
        folds=(0,),
        cuda_device=3,
        tta=True,
        variance=True,
        mean=True,
        entropy=True
    )

    # mc_inference.run()
    mc_inference.compute_uncertainty()

    mc_inference = MonteCarloInference(
        dataset_name = 'Dataset003_ImageCAS_split',
        model='nnUNetTrainerDropout__p02_s1__3d_fullres',
        n_cases=20,
        n_sim=30,
        folds=(0,),
        cuda_device=3,
        tta=True,
        variance=True,
        mean=True,
        entropy=True
    )

    # mc_inference.run()
    mc_inference.compute_uncertainty()

    mc_inference = MonteCarloInference(
        dataset_name = 'Dataset003_ImageCAS_split',
        model='nnUNetTrainerDropout__p02_s3__3d_fullres',
        n_cases=20,
        n_sim=30,
        folds=(0,),
        cuda_device=3,
        tta=True,
        variance=True,
        mean=True,
        entropy=True
    )

    # mc_inference.run()
    mc_inference.compute_uncertainty()
   '''