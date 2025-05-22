from nnunetv2.inference.monte_carlo_inference import MonteCarloInference

dataset_name = 'Dataset003_ImageCAS_split'

n_cases = 20

base = MonteCarloInference( # should this use normal nnunet inference?
    dataset_name=dataset_name,
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0,),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)

tta = MonteCarloInference( # should this use normal nnunet inference?
    dataset_name=dataset_name,
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0,),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)

ens = MonteCarloInference( # should this use normal nnunet inference? # !!! turn of tta
    dataset_name=dataset_name,
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)

mcd = MonteCarloInference( # !!! turn of tta
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0,),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)

tta_ens = MonteCarloInference(
    dataset_name=dataset_name,
    model='not_sure, maybe train base resenc',
    n_cases=n_cases,
    n_sim=1,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)

tta_mcd = MonteCarloInference(
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0,),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)

ens_mcd = MonteCarloInference( # !!! turn of tta
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)

all = MonteCarloInference(
    dataset_name=dataset_name,
    model='s2p02',
    n_cases=n_cases,
    n_sim=30,
    folds=(0, 1, 2, 3, 4),
    cuda_device=0,
    variance=True,
    mean=True,
    entropy=True
)