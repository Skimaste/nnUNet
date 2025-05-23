from nnunetv2.inference.monte_carlo_inference import MonteCarloInference

dataset_name = 'Dataset003_ImageCAS_split'

n_cases = 20 # maybe 100 for final results??

# todo: assign gpu cores depending on runtime

'''
script for comparing the different model combinations for uncertainty estimation
- base: base model
- tta: base model with tta
- ens: ensemble of 5 models
- mcd: mcdropout
- tta_ens: ensemble of 5 models with tta
- tta_mcd: mcdropout with tta
- ens_mcd: ensemble of 5 models with mcdropout
- all: ensemble of 5 models with tta and mcdropout
'''

base = MonteCarloInference( # should this use normal nnunet inference?
    dataset_name=dataset_name,
    model='nnUNetTrainerDropout__p00_s2__3d_fullres',
    n_cases=n_cases,
    n_sim=1,
    folds=(0,),
    cuda_device=0,
    config_name='base',
    tta=False
) # wait for model to finish training

tta = MonteCarloInference( # should this use normal nnunet inference?
    dataset_name=dataset_name,
    model='nnUNetTrainerDropout__p00_s2__3d_fullres',
    n_cases=n_cases,
    n_sim=1,
    folds=(0,),
    cuda_device=0,
    config_name='tta',
    tta=True
) # wait for model to finish training

ens = MonteCarloInference( # should this use normal nnunet inference? 
    dataset_name=dataset_name,
    model='nnUNetTrainerDropout__p00_s2__3d_fullres',
    n_cases=n_cases,
    n_sim=1,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    config_name='ens',
    tta=False
) # wait for model to finish training

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

tta_ens = MonteCarloInference( # should this use normal nnunet inference? 
    dataset_name=dataset_name,
    model='nnUNetTrainerDropout__p00_s2__3d_fullres',
    n_cases=n_cases,
    n_sim=1,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    config_name='tta_ens',
    tta=True
) # wait for model to finish training

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

ens_mcd = MonteCarloInference( # done
    dataset_name=dataset_name,
    model='nnUNetTrainerDropout__p02_s2__3d_fullres',
    n_cases=n_cases,
    n_sim=30,
    folds=(0, 1, 2, 3, 4),
    cuda_device=2,
    config_name='ens_mcd',
    tta=False
)

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

if __name__ == '__main__':
    base.calc_time()
    tta.calc_time()
    ens.calc_time()
    mcd.calc_time()
    tta_ens.calc_time()
    tta_mcd.calc_time()
    ens_mcd.calc_time()
    all.calc_time()