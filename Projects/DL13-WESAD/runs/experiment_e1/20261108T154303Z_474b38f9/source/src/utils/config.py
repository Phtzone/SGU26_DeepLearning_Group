"""Configuration and project-relative paths, independent of the caller's cwd."""
from pathlib import Path
import copy
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(path=None):
    config = yaml.safe_load((ROOT / 'configs/default.yaml').read_text(encoding='utf-8'))
    if path:
        config = merge(config, yaml.safe_load((ROOT / path).read_text(encoding='utf-8')))
    if config['experiment']['status'] == 'planned':
        raise NotImplementedError('This experiment is planned; only E1 is implemented.')
    if config['data']['max_windows_per_subject'] is not None:
        raise ValueError('Research runs must use every valid window.')
    if config['split']['protocol'] != 'grouped_subject_wise_5fold' or config['split']['split_seed'] != 42:
        raise ValueError('The official protocol requires the fixed seed42 five-fold manifest.')
    if config['experiment']['model'] != 'cnn1d' or config['selection'] != {'metric': 'val_macro_f1', 'mode': 'max', 'tie_breaker': 'val_loss'}:
        raise ValueError('Unsupported model/selection protocol.')
    if config['data']['labels'] != [1, 2, 3] or config['runtime']['parallel_folds'] != 1:
        raise ValueError('E1 uses labels 1/2/3 and sequential folds.')
    if config['training']['optimizer'] != 'adam' or config['training']['loss'] != 'weighted_cross_entropy':
        raise ValueError('Unsupported optimizer or loss.')
    if not all(config['logging'][key] for key in ('log_every_iteration', 'log_every_epoch', 'flush_each_iteration')):
        raise ValueError('Complete live logging is mandatory.')
    tc = config['training']
    if not tc['seeds'] or len(set(tc['seeds'])) != len(tc['seeds']) or not all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in tc['seeds']):
        raise ValueError('Training seeds must be distinct uint32 integers.')
    if any(tc[key] <= 0 for key in ('max_epochs', 'batch_size', 'learning_rate', 'early_stopping_patience', 'gradient_clip_norm')) or tc['early_stopping_min_delta'] < 0 or tc['weight_decay'] < 0:
        raise ValueError('Invalid training hyperparameters.')
    sensors = config['data']['sensors']
    if not sensors or len(set(sensors)) != len(sensors) or not set(sensors) <= {'ACC', 'BVP', 'EDA'}:
        raise ValueError('Unsupported or duplicate sensors.')
    from src.utils.tensorboard_tracker import tracking_options
    tracking_options(config)
    return config


def same_training_config(left, right):
    """Monitoring can change on resume; computation, selection and splits cannot."""
    return {k: v for k, v in left.items() if k not in ('logging', 'tracking')} == {k: v for k, v in right.items() if k not in ('logging', 'tracking')}


def raw_root():
    return Path(os.environ.get('WESAD_ROOT', ROOT / 'data/raw/WESAD')).resolve()
