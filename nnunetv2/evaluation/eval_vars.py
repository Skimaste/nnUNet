from nnunetv2.evaluation.evaluation import Evaluater

if __name__ == "__main__":
    label_dir = "/data/dsstu/nnUNet_raw/Dataset003_ImageCAS_split/labelsTs"

    result_root = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s2__3d_fullres/inference/all"
    evaluator = Evaluater(result_root, label_dir, n_cases=20)
    evaluator.run_evaluation()

    result_root = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s2__3d_fullres/inference/ens_mcd"
    evaluator = Evaluater(result_root, label_dir, n_cases=20)
    evaluator.run_evaluation()

    result_root = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s2__3d_fullres/inference/mcd"
    evaluator = Evaluater(result_root, label_dir, n_cases=20)
    evaluator.run_evaluation()

    result_root = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s2__3d_fullres/inference/tta_mcd"
    evaluator = Evaluater(result_root, label_dir, n_cases=20)
    evaluator.run_evaluation()