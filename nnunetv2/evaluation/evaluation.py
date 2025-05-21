import os
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.classification import MulticlassCalibrationError

class Evaluater:
    def __init__(self, result_root, label_dir, threshold=0.5, bins=10, n_cases=10):
        
        self.result_root = result_root
        self.label_dir = label_dir
        self.threshold = threshold
        self.bins = bins
        self.summary = {}
        self.n_cases = n_cases

    def evaluate_case(self, prob_map, variance_map, gt_map, shannon_map):
        prob_map = np.moveaxis(prob_map, 0, -1)  # (H, W, D, C)
        prob = prob_map.reshape(-1, prob_map.shape[-1])
        var = variance_map.flatten()
        gt = gt_map.flatten()
        shannon = shannon_map.flatten()
        # Normalize uncertainty
        var = (var - np.min(var)) / (np.max(var) - np.min(var))
        entropy = (shannon - np.min(shannon)) / (np.max(shannon) - np.min(shannon))

        # Convert to tensors
        prob = torch.tensor(prob, dtype=torch.float32)
        var = torch.tensor(var, dtype=torch.float32)
        gt = torch.tensor(gt, dtype=torch.float32)
        entropy = torch.tensor(entropy, dtype=torch.float32)

        
        eps = 1e-3
        # remove all of this if you have more than 2 classes or you dont want to remove zeros
        # creating masks to ignore zero values
        mask_var = var > eps
        mask_entropy = entropy > eps
        mask_prob0 = prob[:, 0] > eps
        mask_prob1 = prob[:, 1] > eps
        mask_prob = mask_prob0 & mask_prob1  # Combine masks for both classes
        

        # Filter out zero values
        gt_var = gt[mask_var]
        gt_entropy = gt[mask_entropy]
        var_var = var[mask_var]
        entropy_entropy = entropy[mask_entropy]
        prob_prob = prob[mask_prob]
        prob_var = prob[:,1][mask_var]
        prob_entropy = prob[:,1][mask_entropy]
        gt_prob = gt[mask_prob]

       

        # ECE
        ece_metric = MulticlassCalibrationError(num_classes=prob.shape[-1], n_bins=self.bins)
        ece_prob = ece_metric(prob_prob, gt_prob).item()

        # EUCE
        
        pred_var = (prob_var >= self.threshold).float()
        correct_var = (pred_var == gt_var).float()

        ece_variance = 0.0
        for i in range(self.bins):
            bin_lower = i / self.bins
            bin_upper = (i + 1) / self.bins
            mask = (var_var >= bin_lower) & (var_var < bin_upper)
            if torch.any(mask):
                bin_err = 1 - torch.mean(correct_var[mask])
                bin_conf = torch.mean(var_var[mask])
                bin_weight = torch.sum(mask).item() / len(prob_var)
                ece_variance += bin_weight * abs(bin_err.item() - bin_conf.item())

       
        pred_entropy = (prob_entropy >= self.threshold).float()
        correct_entropy = (pred_entropy == gt_entropy).float()
        
        ece_entropy = 0.0
        for i in range(self.bins):
            bin_lower = i / self.bins
            bin_upper = (i + 1) / self.bins
            mask = (entropy_entropy >= bin_lower) & (entropy_entropy < bin_upper)
            if torch.any(mask):
                bin_err = 1 - torch.mean(correct_entropy[mask])
                bin_conf = torch.mean(entropy_entropy[mask])
                bin_weight = torch.sum(mask).item() / len(prob_entropy)
                ece_entropy += bin_weight * abs(bin_err.item() - bin_conf.item())


        return ece_prob, ece_variance, ece_entropy

    def compute_DSC(self, prob_map, gt_map):
        prob_map = np.moveaxis(prob_map, 0, -1)
        pred = np.argmax(prob_map, axis=-1)
        pred = pred.flatten()

        gt = gt_map.flatten()

        # Compute DSC
        intersection = np.sum((pred == 1) & (gt == 1))
        dsc = (2 * intersection) / (np.sum(pred == 1) + np.sum(gt == 1))
        return dsc
    
    def compute_HD95(self, prob_map, gt_map):
        return None


    def run_evaluation(self):
        results = []
        n_cases_evaluated = 0

        # Get list of all case directories in result_root
        case_dirs = [os.path.join(self.result_root, case) for case in os.listdir(self.result_root) if os.path.isdir(os.path.join(self.result_root, case))]

        # Limit the number of cases if n_cases is provided
        if self.n_cases is not None:
            case_dirs = case_dirs[:self.n_cases]
        
        # Iterate over the selected case directories
        for case_dir in case_dirs:
            case = os.path.basename(case_dir)
            try:
                prob_path = os.path.join(case_dir, f"{case}_mean.nii.gz")
                var_path = os.path.join(case_dir, f"{case}_var.nii.gz")
                gt_path = os.path.join(self.label_dir, f"{case}.nii.gz")
                entropy_path = os.path.join(case_dir, f"{case}_shannon_entropy.nii.gz")

                # Check if all necessary files exist
                if not (os.path.exists(prob_path) and os.path.exists(var_path) and os.path.exists(gt_path) and os.path.exists(entropy_path)):
                    print(f"Missing files for {case}. Skipping.")
                    continue

                # Load the data
                prob_map = nib.load(prob_path).get_fdata()
                var_map = nib.load(var_path).get_fdata()
                gt_map = nib.load(gt_path).get_fdata()
                entropy_map = nib.load(entropy_path).get_fdata()

                # Evaluate the case
                ece_prob, ece_variance, ece_entropy = self.evaluate_case(prob_map, var_map, gt_map, entropy_map)

                # Store the result
                results.append({
                    "case": case,
                    "ECE for probabilities": ece_prob,
                    "ECE for variance": ece_variance,
                    "ECE for entropy": ece_entropy
                })
                print(f"{case} -> ECE for probabilities: {ece_prob:.5f}, ECE for variance: {ece_variance:.5f}, ECE for Shannon entropy: {ece_entropy:.5f}")
                n_cases_evaluated += 1  # Increment the number of cases evaluated
            except Exception as e:
                print(f"Error in case {case}: {e}")

        # Save results to a JSON file
        self.save_summary(results, n_cases_evaluated)


    def save_summary(self, results, n_cases_evaluated):
        if results:
            ece_prob_vals = [r["ECE for probabilities"] for r in results]
            ece_variance_vals = [r["ECE for variance"] for r in results]
            ece_shannon_vals = [r["ECE for entropy"] for r in results]

            mean_summary = {
                "mean ECE for probabilities": float(np.mean(ece_prob_vals)),
                "mean ECE for variance": float(np.mean(ece_variance_vals)),
                "mean ECE for entropy": float(np.mean(ece_shannon_vals))
            }
        else:
            mean_summary = {
                "mean_ECE for probabilities": None,
                "mean ECE for variance": None,
                "mean ECE for entropy": None
            }

        summary = {
            "number_of_cases_evaluated": n_cases_evaluated,
            "mean_metrics": mean_summary,
            "results": results
        }

        json_path = os.path.join(self.result_root, "evaluation_summary_masked.json")
        with open(json_path, "w") as jsonfile:
            json.dump(summary, jsonfile, indent=4)

        print(f"\nSaved evaluation summary to: {json_path}")


if __name__ == "__main__":
    
    result_root_1 = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s2__3d_fullres/inference"
    result_root_2 = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p01_s2__3d_fullres/inference"
    result_root_3 = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p05_s2__3d_fullres/inference"
    result_root_4 = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s1__3d_fullres/inference"
    result_root_5 = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s3__3d_fullres/inference"
    label_dir = "/data/dsstu/nnUNet_raw/Dataset003_ImageCAS_split/labelsTs"

    evaluator_1 = Evaluater(result_root_1, label_dir, n_cases=20)
    evaluator_1.run_evaluation()
    evaluator_2 = Evaluater(result_root_2, label_dir, n_cases=20)
    evaluator_2.run_evaluation()
    evaluator_3 = Evaluater(result_root_3, label_dir, n_cases=20)
    evaluator_3.run_evaluation()
    evaluator_4 = Evaluater(result_root_4, label_dir, n_cases=20)
    evaluator_4.run_evaluation()
    evaluator_5 = Evaluater(result_root_5, label_dir, n_cases=20)
    evaluator_5.run_evaluation()