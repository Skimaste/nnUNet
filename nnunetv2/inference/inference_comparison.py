from nnunetv2.inference.monte_carlo_inference import MonteCarloInference

dataset_name = 'Dataset003_ImageCAS_split'

n_cases = 20

base = MonteCarloInference(
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

tta = MonteCarloInference(
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