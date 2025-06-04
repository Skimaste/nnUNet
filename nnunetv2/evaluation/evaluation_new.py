from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import (
    load_json, join, isdir, listdir, save_json
)
import nibabel as nib
import torch
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
        self.ground_truth_dir = join(nnUNet_raw, dataset, 'imagesTs')
        self.predictions_dir = join(nnUNet_results, dataset, model_dir)
        self.output_dir = join(self.predictions_dir, 'evaluation_results')
        self.gpu_device = (torch.device(gpu_device) if gpu_device is not None else None)
        
        self.results = {}
        self.masking_threshold = masking_threshold
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


    def calc_metrics(self, gt, pred, spacing=None):
        hd95 = self.hausdorff_metric(y_pred=pred, y=gt, spacing=spacing).aggregate().item()
        dice = self.dice_metric(y_pred=pred, y=gt).aggregate().item()
        ece = self.calibration_metric.update(pred, gt.squeeze(1)).compute().item()
        self.hausdorff_metric.reset()
        self.calibration_metric.reset()
        self.dice_metric.reset()

        return hd95, dice, ece
    
    def mask_image(self, gt, pred):
        # Apply the masking threshold to the ground truth and prediction
        mask = pred > self.masking_threshold
        gt_masked = gt[mask]
        pred_masked = pred[mask]
        return gt_masked, pred_masked

    def load_image(self, image_path):
        # Load the image using nibabel
        if not isdir(image_path):
            raise ValueError(f"Provided path {image_path} is not a directory.")
        if not listdir(image_path):
            raise ValueError(f"No images found in the directory {image_path}.")
        img = nib.load(image_path)
        img_data = img.get_fdata()
        return torch.tensor(img_data, dtype=torch.float32).unsqueeze(0), img.affine
    
    def save_results(self, results):
        # Save the results to a JSON file
        output_file = join(self.output_dir, 'evaluation_results.json')
        save_json(results, output_file)
        print(f"Results saved to {output_file}")

    def evaluate(self):
        if not isdir(self.ground_truth_dir) or not isdir(self.predictions_dir):
            raise ValueError("Ground truth and predictions directories must exist.")

        gt_files = listdir(self.ground_truth_dir)
        pred_files = listdir(self.predictions_dir)

        if len(gt_files) != len(pred_files):
            raise ValueError("Number of ground truth files does not match number of prediction files.")

        for gt_file, pred_file in zip(gt_files, pred_files):
            gt_path = join(self.ground_truth_dir, gt_file)
            pred_path = join(self.predictions_dir, pred_file)

            gt_image, spacing = self.load_image(gt_path)
            pred_image, _ = self.load_image(pred_path)

            # Apply masking
            gt_masked, pred_masked = self.mask_image(gt_image, pred_image)

            # Calculate metrics
            hd95, dice, ece = self.calc_metrics(gt_masked, pred_masked, spacing=spacing)

            # Store results
            self.results[gt_file] = {
                'Hausdorff95': hd95,
                'Dice': dice,
                'ECE': ece
            }

        # Save results to output directory
        self.save_results(self.results)

    
        

