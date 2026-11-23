"""Construct implemented models without changing the historical CNN layout."""
from src.models.cnn import CNN1D
from src.models.gru import GRUClassifier


def build_model(config, input_channels):
    name = config['experiment']['model']
    if name == 'cnn1d':
        return CNN1D(input_channels, **config.get('model', {}))
    if name == 'gru':
        return GRUClassifier(input_channels, **config.get('model', {}))
    raise ValueError(f'Unsupported model: {name}')
