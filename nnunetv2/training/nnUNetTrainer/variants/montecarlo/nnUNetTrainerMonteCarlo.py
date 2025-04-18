from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerMonteCarlo(nnUNetTrainer):
    """
    This trainer is used for Monte Carlo dropout. It is a variant of the nnUNetTrainer that uses dropout during
    inference to obtain uncertainty estimates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_monte_carlo = True
        self._num_monte_carlo_samples = kwargs.get('num_monte_carlo_samples', 10)
        self._monte_carlo_dropout = kwargs.get('monte_carlo_dropout', 0.5)
        self._monte_carlo_dropout_mode = kwargs.get('monte_carlo_dropout_mode', 'train')
        self._monte_carlo_dropout_layers = kwargs.get('monte_carlo_dropout_layers', None)
        self._monte_carlo_dropout_kwargs = kwargs.get('monte_carlo_dropout_kwargs', {})
        self._monte_carlo_dropout_kwargs['p'] = self._monte_carlo_dropout
        self._monte_carlo_dropout_kwargs['inplace'] = True
        self._monte_carlo_dropout_kwargs['mode'] = self._monte_carlo_dropout_mode
        self._monte_carlo_dropout_kwargs['layers'] = self._monte_carlo_dropout_layers
        self._monte_carlo_dropout_kwargs['num_samples'] = self._num_monte_carlo_samples
        self._monte_carlo_dropout_kwargs['enable_monte_carlo'] = self._enable_monte_carlo
        self._monte_carlo_dropout_kwargs['monte_carlo_dropout'] = self._monte_carlo_dropout
        self._monte_carlo_dropout_kwargs['monte_carlo_dropout_mode'] = self._monte_carlo_dropout_mode
        self._monte_carlo_dropout_kwargs['monte_carlo_dropout_layers'] = self._monte_carlo_dropout_layers
        self._monte_carlo_dropout_kwargs['monte_carlo_dropout_kwargs'] = self._monte_carlo_dropout_kwargs