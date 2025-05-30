from nnunetv2.inference.monte_carlo_inference import MonteCarloInference

if __name__ == '__main__':
    dataset_name = 'Dataset003_ImageCAS_split'

    n_cases = 20 # maybe 100 for final results??

    base = MonteCarloInference( # done
        dataset_name=dataset_name,
        model='nnUNetTrainerDropout__p00_s2__3d_fullres',
        n_cases=n_cases,
        n_sim=1,
        folds=(0,),
        cuda_device=3,
        config_name='base',
        tta=False
    )

    base.run()

    tta = MonteCarloInference( # done
        dataset_name=dataset_name,
        model='nnUNetTrainerDropout__p00_s2__3d_fullres',
        n_cases=n_cases,
        n_sim=1,
        folds=(0,),
        cuda_device=3,
        config_name='tta',
        tta=True
    )

    tta.run()

    ens = MonteCarloInference( # done
        dataset_name=dataset_name,
        model='nnUNetTrainerDropout__p00_s2__3d_fullres',
        n_cases=n_cases,
        n_sim=1,
        folds=(0, 1, 2, 3, 4),
        cuda_device=3,
        config_name='ens',
        tta=False
    )

    ens.run()

    tta_ens = MonteCarloInference( # done
        dataset_name=dataset_name,
        model='nnUNetTrainerDropout__p00_s2__3d_fullres',
        n_cases=n_cases,
        n_sim=1,
        folds=(0, 1, 2, 3, 4),
        cuda_device=3,
        config_name='tta_ens',
        tta=True
    )

    tta_ens.run()

