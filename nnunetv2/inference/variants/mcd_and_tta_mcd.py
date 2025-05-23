from nnunetv2.inference.monte_carlo_inference import MonteCarloInference

if __name__ == '__main__':
    dataset_name = 'Dataset003_ImageCAS_split'

    n_cases = 20 # maybe 100 for final results??

    mcd = MonteCarloInference( # done
        dataset_name=dataset_name,
        model='nnUNetTrainerDropout__p02_s2__3d_fullres',
        n_cases=n_cases,
        n_sim=30,
        folds=(0,),
        cuda_device=0,
        config_name='mcd',
        tta=False
    )

    mcd.run()

    tta_mcd = MonteCarloInference( # done
        dataset_name=dataset_name,
        model='nnUNetTrainerDropout__p02_s2__3d_fullres',
        n_cases=n_cases,
        n_sim=30,
        folds=(0,),
        cuda_device=0,
        config_name='tta_mcd',
        tta=True
    )

    tta_mcd.run()

