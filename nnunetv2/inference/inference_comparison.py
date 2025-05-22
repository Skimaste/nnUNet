from nnunetv2.inference.monte_carlo_inference import MonteCarloInference

dataset_name = 'Dataset003_ImageCAS_split'

n_cases = 20

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
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0,),
    cuda_device=0,
    tta=False
)

tta = MonteCarloInference( # should this use normal nnunet inference?
    dataset_name=dataset_name,
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0,),
    cuda_device=0,
    tta=True
)

ens = MonteCarloInference( # should this use normal nnunet inference? 
    dataset_name=dataset_name,
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    tta=False
)

mcd = MonteCarloInference( # done
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0,),
    cuda_device=0,
    tta=False
)

tta_ens = MonteCarloInference( # should this use normal nnunet inference? 
    dataset_name=dataset_name,
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    tta=True
)

tta_mcd = MonteCarloInference( # done
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0,),
    cuda_device=0,
    tta=True
)

ens_mcd = MonteCarloInference( # done
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    tta=False
)

all = MonteCarloInference( # done
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    tta=True
)