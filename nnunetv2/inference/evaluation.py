import numpy as np
import os
import shutil
import nibabel as nib
import pickle


file_path_prob = '/data/dsstu/nnUNet_raw/Dataset003_ImageCAS_split/case_0015_mean.nii.gz'
file_path_unc = '/data/dsstu/nnUNet_raw/Dataset003_ImageCAS_split/case_0015_variance.nii.gz'
file_path_gt = '/data/dsstu/nnUNet_raw/Dataset003_ImageCAS_split/labelsTs/case_0015.nii.gz'

probabibilty_map = nib.load(file_path_prob).get_fdata()
#print("Shape:", probabibilty_map.shape)

uncertainty_map = nib.load(file_path_unc).get_fdata()
#print("Shape:", uncertainty_map.shape)

ground_truth = nib.load(file_path_gt).get_fdata()
#print("Shape:", ground_truth.shape)

# Change for multicalss implementation and multiple threads

def evaluate(probabibilty_map, uncertainty_map, ground_truth, threshold=0.5, bins=10):

    prob = probabibilty_map.flatten()
    unc = uncertainty_map.flatten()
    gt = ground_truth.flatten()

    pred = (prob > threshold).astype(np.float32)
    correct = (pred == gt).astype(np.float32)

    norm_unc = (unc - np.min(unc)) / (np.max(unc) - np.min(unc))

    ece = 0.0
    euce = 0.0

    bin_edges = np.linspace(0, 1, bins + 1)
    for i in range(bins): 
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        mask = (prob >= bin_lower) & (prob < bin_upper)

        if np.any(mask):
            bin_acc = np.mean(correct[mask])
            if bin_upper <= threshold:
                bin_conf = 1-np.mean(prob[mask])
            else:
                bin_conf = np.mean(prob[mask])
            ece += (np.sum(mask) / len(prob)) * np.abs(bin_acc - bin_conf)
            print(f'Bin{i} sum mask: {np.sum(mask)}')
            print(f'Bin{i} len prob: {np.sum(len(prob))}')
            print(f'Bin{i} bin acc: {bin_acc}')
            print(f'Bin{i} bin conf: {bin_conf}')

    for i in range(bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        mask = (norm_unc >= bin_lower) & (norm_unc < bin_upper)
        if np.any(mask):
            bin_err = 1 - np.mean(correct[mask])
            bin_conf = np.mean(norm_unc[mask])
            euce += (np.sum(mask) / len(prob)) * np.abs(bin_err - bin_conf)

    for i in range(bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        mask = (prob >= bin_lower) & (prob < bin_upper)
        print(f"Bin {i}: {np.sum(mask)} samples")

    return ece, euce

ece, euce = evaluate(probabibilty_map, uncertainty_map, ground_truth, threshold=0.5, bins=10)
print("ECE:", ece)
print("EUCE:", euce)

  