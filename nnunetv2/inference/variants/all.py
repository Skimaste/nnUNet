from nnunetv2.inference.monte_carlo_inference import MonteCarloInference

if __name__ == '__main__':
    dataset_name = 'Dataset003_ImageCAS_split'

    n_cases = 20 # maybe 100 for final results??

    all = MonteCarloInference( # done
        dataset_name=dataset_name,
        model='nnUNetTrainerDropout__p02_s2__3d_fullres',
        n_cases=n_cases,
        n_sim=30,
        folds=(0, 1, 2, 3, 4),
        cuda_device=3,
        config_name='all',
        tta=True
    )

    all.run()
