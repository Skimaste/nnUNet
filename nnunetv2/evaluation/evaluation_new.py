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

    
    def mae_metric(self, y_pred, y, y_ent):
        """
        Calculate Mean Absolute Error (MAE) between entropy maps and ground truth minus prediction.
        """
        return torch.mean(torch.abs(torch.abs(y_pred - y) - y_ent))


    def calc_metrics(self, gt, pred, pred_seg, entropy, spacing=None):

        # hd95
        self.hausdorff_metric(y_pred=pred_seg, y=gt, spacing=spacing)
        hd95 = self.hausdorff_metric.aggregate().item()
        self.hausdorff_metric.reset()

        # dice
        self.dice_metric(y_pred=pred_seg, y=gt)
        dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        # ECE
        self.calibration_metric.update(pred, gt.squeeze(1))
        ece = self.calibration_metric.compute().item()
        self.calibration_metric.reset()

        # MAE
        mae = self.mae_metric(y_pred=pred, y=gt, y_ent=entropy).item()

        return hd95, dice, ece, mae


    def mask_image(self, mask_from, *args):
        # Apply the masking from the first image to all subsequent images
        mask = mask_from > self.masking_threshold
        masked_images = []
        for image in args:
            if image.shape != mask_from.shape:
                raise ValueError("All images must have the same shape for masking.")
            masked_image = image[mask]
            masked_images.append(masked_image)
        if len(masked_images) == 1:
            return masked_images[0], None
        return masked_images


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
        ent_files = listdir(self.predictions_dir) # !!! what are these called?

        if len(gt_files) != len(pred_files):
            raise ValueError("Number of ground truth files does not match number of prediction files.")

        for gt_file, pred_file, ent_file in zip(gt_files, pred_files, ent_files):
            gt_path = join(self.ground_truth_dir, gt_file)
            pred_path = join(self.predictions_dir, pred_file)
            ent_path = join(self.predictions_dir, ent_file)

            gt_image, spacing = self.load_image(gt_path)
            pred_image, _ = self.load_image(pred_path)
            ent_image, _ = self.load_image(ent_path)

            # put the images on the GPU if specified
            if self.gpu_device is not None:
                gt_image = gt_image.to(self.gpu_device)
                pred_image = pred_image.to(self.gpu_device)
                ent_image = ent_image.to(self.gpu_device)

            # Apply masking
            gt_masked, pred_masked = self.mask_image(gt_image, pred_image)

            # Calculate metrics
            hd95, dice, ece, mae = self.calc_metrics(gt_masked, pred_masked, entropy=entropy, spacing=spacing)

            # Store results
            self.results[gt_file] = {
                'Hausdorff95': hd95,
                'Dice': dice,
                'ECE': ece,
                'MAE': mae
            }

        self.results['summary'] = {
            'Hausdorff95': torch.tensor([res['Hausdorff95'] for res in self.results.values() if 'Hausdorff95' in res]).mean().item(),
            'Dice': torch.tensor([res['Dice'] for res in self.results.values() if 'Dice' in res]).mean().item(),
            'ECE': torch.tensor([res['ECE'] for res in self.results.values() if 'ECE' in res]).mean().item(),
            'MAE': torch.tensor([res['MAE'] for res in self.results.values() if 'MAE' in res]).mean().item()
        }

        # Save results to output directory
        self.save_results(self.results)


    def test_run(self):
        print("Starting evaluation...")
        self.evaluate()
        print("Evaluation completed successfully.")


    def run(self):
        self.evaluate()
        


if __name__ == '__main__':

    '''
    # Example usage
    gt = torch.randint(0, 2, (1, 1, 64, 64, 64))  # Example ground truth tensor
    pred = torch.softmax(torch.rand(1, 2, 64, 64, 64), dim=1)  # Example prediction tensor
    pred_seg = torch.argmax(pred, dim=1, keepdim=True)  # Convert to segmentation

    def entropy_from_softmax(pred):
        return -torch.sum(pred * torch.log(pred + 1e-10), dim=1, keepdim=True)
    entropy = entropy_from_softmax(pred)  # Calculate entropy from softmax predictions
    # entropy = torch.rand(1, 1, 64, 64, 64)  # Example entropy tensor

    # print shapes of tensors
    print(f"GT shape: {gt.shape}, Pred shape: {pred.shape}, Entropy shape: {entropy.shape}")

    eval = Evaluator('none', 'none', gpu_device=2)

    for x in (gt, pred, entropy):
        x.to(eval.gpu_device)

    print(eval.calc_metrics(gt, pred, pred_seg, entropy))

    '''

    eval = Evaluator('Dataset003_ImageCAS_split', 'nnUNetTrainerDropout__p00_s2__3d_fullres/base', gpu_device=2)

    # print directories used by evaluator
    print(f"Ground truth directory: {eval.ground_truth_dir}")
    print(f"Predictions directory: {eval.predictions_dir}")
    # print output directory
    print(f"Output directory: {eval.output_dir}")

    # eval.test_run()
