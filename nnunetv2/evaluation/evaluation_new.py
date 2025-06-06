from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import (
    load_json, join, isdir, listdir, save_json, maybe_mkdir_p
)
import nibabel as nib
import torch
import re
import numpy as np
import gc

from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from torchmetrics.classification import MulticlassCalibrationError

class Evaluator:
    def __init__(self, dataset, model_dir, gpu_device=None,
                 calibration_metric_num_classes=2, calibration_metric_n_bins=10,
                 hd_distance_include_background=False, hd_distance_metric='euclidean',
                 hd_distance_percentile=95,
                 dice_include_background=False, dice_reduction='mean',
                 masking_threshold=0.1
                 ):
        self.dataset = dataset
        self.ground_truth_dir = join(nnUNet_raw, dataset, 'labelsTs')
        self.predictions_dir = join(nnUNet_results, dataset, model_dir)
        self.output_dir = join(self.predictions_dir, 'evaluation_results')
        self.gpu_device = (torch.device(gpu_device) if gpu_device is not None else None)
        
        self.results = {}
        self.masking_threshold = masking_threshold

        # Initialize metrics
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=hd_distance_include_background, 
            distance_metric=hd_distance_metric, 
            percentile=hd_distance_percentile
            )
        self.calibration_metric = MulticlassCalibrationError(
            num_classes=calibration_metric_num_classes, 
            n_bins=calibration_metric_n_bins
            )
        self.dice_metric = DiceMetric(
            include_background=dice_include_background, 
            reduction=dice_reduction
            )


    # metrics
    def hd95(self, y_pred, y, spacing=None):
        """
        Calculate the 95th percentile of the Hausdorff distance.
        """
        self.hausdorff_metric(y_pred=y_pred, y=y, spacing=spacing)
        hd95 = self.hausdorff_metric.aggregate().item()
        self.hausdorff_metric.reset()
        return hd95
    

    def dice(self, y_pred, y):
        """
        Calculate the Dice coefficient.
        """
        self.dice_metric(y_pred=y_pred, y=y)
        dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        return dice
    

    def ece(self, y_pred, y):
        """
        Calculate the Expected Calibration Error (ECE).
        """
        self.calibration_metric.update(y_pred, y)
        ece = self.calibration_metric.compute().item()
        self.calibration_metric.reset()
        return ece
    

    def mae(self, y_ent, y_pred, y):
        """
        Calculate the Mean Absolute Error (MAE).
        """
        error = torch.abs(y_pred - y)
        mae = torch.mean(torch.abs(error - y_ent))
        return mae.item()


    def mask_image(self, mask_from, *args):
        # Apply the masking from the first image to all subsequent images
        mask = mask_from > self.masking_threshold
        masked_images = [mask_from[mask]]
        for image in args:
            if image.shape != mask_from.shape:
                raise ValueError("All images must have the same shape for masking.")
            masked_image = image[mask]
            masked_images.append(masked_image)
        return masked_images


    def load_image(self, image_path, is_label=False):
        # Load the image using nibabel
        img = nib.load(image_path)
        img_data = img.get_fdata()
        spacing = np.abs(img.affine.diagonal()[:3]) # Get spacing from the affine matrix
        if is_label:
            return torch.tensor(img_data, dtype=torch.long).unsqueeze(0), spacing
        return torch.tensor(img_data, dtype=torch.float32).unsqueeze(0), spacing


    def save_results(self, results):
        # Save the results to a JSON file
        output_file = join(self.output_dir, 'evaluation_summary.json')
        save_json(results, output_file)
        print(f"Results saved to {output_file}")


    def find_cases(self):
        cases = listdir(self.predictions_dir)
        cases = [case for case in cases if re.match(r'case_\d+', case)]
        return cases


    def evaluate(self):
        cases = self.find_cases()

        for case in cases:  # For testing, limit to first 2 cases
            print(f"Evaluating {case}")

            gt_path = join(self.ground_truth_dir, f'{case}.nii.gz')
            pred_path = join(self.predictions_dir, case, f'{case}_mean.nii.gz')
            ent_path = join(self.predictions_dir, case, f'{case}_shannon_entropy.nii.gz')

            gt_image, spacing = self.load_image(gt_path, is_label=True)  # Load ground truth as label
            prob_image, _ = self.load_image(pred_path)
            ent_image, _ = self.load_image(ent_path)

            # put the images on the GPU if specified
            if self.gpu_device is not None:
                gt_image = gt_image.to(self.gpu_device) # shape [1, 512, 512, z]
                prob_image = prob_image.to(self.gpu_device) # shape [1, 2, 512, 512, z]
                ent_image = ent_image.to(self.gpu_device) # shape [1, 512, 512, z]

            seg_binary = torch.argmax(prob_image, dim=1)  # shape [1, 512, 512, z]
            seg_onehot_class_last = torch.nn.functional.one_hot(seg_binary, num_classes=prob_image.shape[1])  # shape [1, 512, 512, z, 2]
            seg_onehot = seg_onehot_class_last.permute(0, 4, 1, 2, 3)  # shape [1, 2, 512, 512, z]

            # print(f'seg_image shape: {seg_binary.shape}, seg_onehot shape: {seg_onehot.shape}, gt_image shape: {gt_image.shape}, prob_image shape: {prob_image.shape}, ent_image shape: {ent_image.shape}')

            # Calculate metrics for segmentation
            hd95 = self.hd95(seg_onehot, gt_image.unsqueeze(0), spacing=spacing)
            dice = self.dice(seg_onehot, gt_image.unsqueeze(0))

            # Calculate metrics for probabilities and entropy
            ece = self.ece(prob_image, gt_image)
            # Apply masking to the images
            # Note: The mask_image function expects the first argument to be the mask image
            # and applies it to all subsequent images.
            # Here, we use prob_image as the mask image
            # and apply it to both prob_image and gt_image.
            images = self.mask_image(prob_image[:, 1, :, :, :], prob_image[:, 0, :, :, :], gt_image)
            prob_masked = torch.stack([images[0], images[1]], dim=1).unsqueeze(0).permute(0, 2, 1) # shape [1, 2, x*y*z[masked]]
            gt_masked = images[2].unsqueeze(0) # shape [1, x*y*z[masked]] 
            ece_masked = self.ece(prob_masked, gt_masked)

            mae = self.mae(ent_image, prob_image[:, 1, :, :, :], gt_image)
            mae_masked = self.mae(*self.mask_image(ent_image, prob_image[:, 1, :, :, :], gt_image)) # Apply masking

            # Store results
            self.results[case] = {
                'Hausdorff95': hd95,
                'Dice': dice,
                'ECE': ece,
                'ECE_masked': ece_masked,
                'MAE': mae,
                'MAE_masked': mae_masked
            }

            print(f'Finished {case}')

            # Clear GPU memory
            
            del gt_image, prob_image, ent_image, seg_binary, seg_onehot_class_last, seg_onehot
            gc.collect()
            if self.gpu_device is not None:
                torch.cuda.empty_cache()
        
        # Calculate summary metrics
        self.results['summary'] = {
            'Hausdorff95': torch.tensor([res['Hausdorff95'] for res in self.results.values() if 'Hausdorff95' in res]).mean().item(),
            'Dice': torch.tensor([res['Dice'] for res in self.results.values() if 'Dice' in res]).mean().item(),
            'ECE': torch.tensor([res['ECE'] for res in self.results.values() if 'ECE' in res]).mean().item(),
            'ECE_masked': torch.tensor([res['ECE_masked'] for res in self.results.values() if 'ECE_masked' in res]).mean().item(),
            'MAE': torch.tensor([res['MAE'] for res in self.results.values() if 'MAE' in res]).mean().item(),
            'MAE_masked': torch.tensor([res['MAE_masked'] for res in self.results.values() if 'MAE_masked' in res]).mean().item()
        }
        
        # Save results to output directory
        maybe_mkdir_p(self.output_dir)
        self.save_results(self.results)


    def run(self):
        self.evaluate()
        


if __name__ == '__main__':

    dataset = 'Dataset003_ImageCAS_split'

    model = 'nnUNetTrainerDropout__p00_s2__3d_fullres'

    eval = Evaluator(dataset, join(model, 'base'), gpu_device=2)
    eval.run()
    
    eval = Evaluator(dataset, join(model, 'ens'), gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, join(model, 'tta'), gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, join(model, 'tta_ens'), gpu_device=2)
    eval.run()

    model = 'nnUNetTrainerDropout__p02_s2__3d_fullres'

    eval = Evaluator(dataset, join(model, 'all'), gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, join(model, 'ens_mcd'), gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, join(model, 'mcd'), gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, join(model, 'tta_mcd'), gpu_device=2)
    eval.run()


    # mc models
    eval = Evaluator(dataset, 'nnUNetTrainerDropout__p02_s2__3d_fullres/inference', gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, 'nnUNetTrainerDropout__p01_s2__3d_fullres/inference', gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, 'nnUNetTrainerDropout__p05_s2__3d_fullres/inference', gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, 'nnUNetTrainerDropout__p02_s1__3d_fullres/inference', gpu_device=2)
    eval.run()

    eval = Evaluator(dataset, 'nnUNetTrainerDropout__p02_s3__3d_fullres/inference', gpu_device=2)
    eval.run()
    