import os
import json
import nibabel as nib
import numpy as np
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
        norm_unc = (var - np.min(var)) / (np.max(var) - np.min(var) + 1e-8)
        norm_shannon = (shannon - np.min(shannon)) / (np.max(shannon) - np.min(shannon) + 1e-8)

        # Convert to tensors
        prob = torch.tensor(prob, dtype=torch.float32)
        norm_unc = torch.tensor(norm_unc, dtype=torch.float32)
        gt = torch.tensor(gt, dtype=torch.float32)
        norm_shannon = torch.tensor(norm_shannon, dtype=torch.float32)

        # ECE
        ece_metric = MulticlassCalibrationError(num_classes=prob.shape[-1], n_bins=self.bins)
        ece_mean = ece_metric(prob, gt).item()

        # EUCE
        prob_class1 = prob[:, 1]
        pred = (prob_class1 >= self.threshold).float()
        correct = (pred == gt).float()

        ece_variance = 0.0
        for i in range(self.bins):
            bin_lower = i / self.bins
            bin_upper = (i + 1) / self.bins
            mask = (norm_unc >= bin_lower) & (norm_unc < bin_upper)
            if torch.any(mask):
                bin_err = 1 - torch.mean(correct[mask])
                bin_conf = torch.mean(norm_unc[mask])
                bin_weight = torch.sum(mask).item() / len(prob)
                ece_variance += bin_weight * abs(bin_err.item() - bin_conf.item())
        
        ece_shannon = 0.0
        for i in range(self.bins):
            bin_lower = i / self.bins
            bin_upper = (i + 1) / self.bins
            mask = (norm_shannon >= bin_lower) & (norm_shannon < bin_upper)
            if torch.any(mask):
                bin_err = 1 - torch.mean(correct[mask])
                bin_conf = torch.mean(norm_shannon[mask])
                bin_weight = torch.sum(mask).item() / len(prob)
                ece_shannon += bin_weight * abs(bin_err.item() - bin_conf.item())


        return ece_mean, ece_variance, ece_shannon

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
                unc_path = os.path.join(case_dir, f"{case}_var.nii.gz")
                gt_path = os.path.join(self.label_dir, f"{case}.nii.gz")
                shannon_path = os.path.join(case_dir, f"{case}_shannon_entropy.nii.gz")

                # Check if all necessary files exist
                if not (os.path.exists(prob_path) and os.path.exists(unc_path) and os.path.exists(gt_path) and os.path.exists(shannon_path)):
                    print(f"Missing files for {case}. Skipping.")
                    continue

                # Load the data
                prob_map = nib.load(prob_path).get_fdata()
                unc_map = nib.load(unc_path).get_fdata()
                gt_map = nib.load(gt_path).get_fdata()
                shannon_map = nib.load(shannon_path).get_fdata()

                # Evaluate the case
                ece_mean, ece_variance, ece_shannon = self.evaluate_case(prob_map, unc_map, gt_map, shannon_map)

                # Store the result
                results.append({
                    "case": case,
                    "ECE for mean": ece_mean,
                    "ECE for variance": ece_variance,
                    "ECE for Shannon entropy": ece_shannon
                })
                print(f"{case} -> ECE for mean: {ece_mean:.5f}, ECE for variance: {ece_variance:.5f}, ECE for Shannon entropy: {ece_shannon:.5f}")
                n_cases_evaluated += 1  # Increment the number of cases evaluated
            except Exception as e:
                print(f"Error in case {case}: {e}")

        # Save results to a JSON file
        self.save_summary(results, n_cases_evaluated)

    def save_summary(self, results, n_cases_evaluated):
        if results:
            ece_mean_vals = [r["ECE for mean"] for r in results]
            ece_variance_vals = [r["ECE for variance"] for r in results]
            ece_shannon_vals = [r["ECE for Shannon entropy"] for r in results]

            mean_summary = {
                "mean_ECE_for_mean": float(np.mean(ece_mean_vals)),
                "mean_ECE_for_variance": float(np.mean(ece_variance_vals)),
                "mean_ECE_for_Shannon_entropy": float(np.mean(ece_shannon_vals))
            }
        else:
            mean_summary = {
                "mean_ECE_for_mean": None,
                "mean_ECE_for_variance": None,
                "mean_ECE_for_Shannon_entropy": None
            }

        summary = {
            "number_of_cases_evaluated": n_cases_evaluated,
            "mean_metrics": mean_summary,
            "results": results
        }

        json_path = os.path.join(self.result_root, "evaluation_summary.json")
        with open(json_path, "w") as jsonfile:
            json.dump(summary, jsonfile, indent=4)

        print(f"\nSaved evaluation summary to: {json_path}")



result_root = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s3__3d_fullres/inference"
label_dir = "/data/dsstu/nnUNet_raw/Dataset003_ImageCAS_split/labelsTs"

evaluator = Evaluater(result_root, label_dir, n_cases=20)
evaluator.run_evaluation()