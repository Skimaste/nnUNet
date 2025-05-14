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

    def evaluate_case(self, prob_map, unc_map, gt_map):
        prob_map = np.moveaxis(prob_map, 0, -1)  # (H, W, D, C)
        prob = prob_map.reshape(-1, prob_map.shape[-1])
        unc = unc_map.flatten()
        gt = gt_map.flatten()

        # Normalize uncertainty
        norm_unc = (unc - np.min(unc)) / (np.max(unc) - np.min(unc) + 1e-8)

        # Convert to tensors
        prob = torch.tensor(prob, dtype=torch.float32)
        norm_unc = torch.tensor(norm_unc, dtype=torch.float32)
        gt = torch.tensor(gt, dtype=torch.float32)

        # ECE
        ece_metric = MulticlassCalibrationError(num_classes=prob.shape[-1], n_bins=self.bins)
        ece = ece_metric(prob, gt).item()

        # EUCE
        prob_class1 = prob[:, 1]
        pred = (prob_class1 >= self.threshold).float()
        correct = (pred == gt).float()

        euce = 0.0
        for i in range(self.bins):
            bin_lower = i / self.bins
            bin_upper = (i + 1) / self.bins
            mask = (norm_unc >= bin_lower) & (norm_unc < bin_upper)
            if torch.any(mask):
                bin_err = 1 - torch.mean(correct[mask])
                bin_conf = torch.mean(norm_unc[mask])
                bin_weight = torch.sum(mask).item() / len(prob)
                euce += bin_weight * abs(bin_err.item() - bin_conf.item())

        return ece, euce

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

                # Check if all necessary files exist
                if not (os.path.exists(prob_path) and os.path.exists(unc_path) and os.path.exists(gt_path)):
                    print(f"Missing files for {case}. Skipping.")
                    continue

                # Load the data
                prob_map = nib.load(prob_path).get_fdata()
                unc_map = nib.load(unc_path).get_fdata()
                gt_map = nib.load(gt_path).get_fdata()

                # Evaluate the case
                ece, euce = self.evaluate_case(prob_map, unc_map, gt_map)

                # Store the result
                results.append({
                    "case": case,
                    "ECE": ece,
                    "EUCE": euce
                })
                print(f"{case} -> ECE: {ece:.5f}, EUCE: {euce:.5f}")
                n_cases_evaluated += 1  # Increment the number of cases evaluated
            except Exception as e:
                print(f"Error in case {case}: {e}")

        # Save results to a JSON file
        self.save_summary(results, n_cases_evaluated)

    def save_summary(self, results, n_cases_evaluated):
        summary = {
            "number_of_cases_evaluated": n_cases_evaluated,
            "results": results
        }
        
        json_path = os.path.join(self.result_root, "evaluation_summary.json")
        with open(json_path, "w") as jsonfile:
            json.dump(summary, jsonfile, indent=4)
        
        print(f"\nSaved evaluation summary to: {json_path}")


result_root = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/mc_n10_p02_s1"
label_dir = "/data/dsstu/nnUNet_raw/Dataset003_ImageCAS_split/labelsTs"

evaluator = Evaluater(result_root, label_dir, n_cases=2)
evaluator.run_evaluation()