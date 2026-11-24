"""Configuration and project-relative paths, independent of the caller's cwd."""
from pathlib import Path
import copy
import os
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]


def experiment_folder(config):
    if config['data'].get('source') == 'common_rate':
        name = config['experiment']['id']
        if not re.fullmatch(r'common_rate64_[a-z0-9_]+', name):
            raise ValueError('Common-rate experiments require a separate common_rate64_* experiment ID.')
        return name
    return {'cnn1d': 'experiment_e1', 'gru': 'experiment_e2'}[config['experiment']['model']]


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(path=None):
    config = yaml.safe_load((ROOT / 'configs/default.yaml').read_text(encoding='utf-8'))
    if path:
        config = merge(config, yaml.safe_load((ROOT / path).read_text(encoding='utf-8')))
    if config['data'].get('input_format') == 'common_rate_windows_v1':
        raise NotImplementedError(
            'The 14-channel common-rate config in configs/ is a superseded local draft. '
            'Use experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml or '
            'experiments/common_rate64_gru_grouped5fold_v1/config.yaml. '
            'Refusing to fall back to historical wrist preprocessing.')
    if config['experiment']['status'] == 'planned':
        raise NotImplementedError('This experiment is planned; only CNN-1D and GRU are implemented.')
    if config['data']['max_windows_per_subject'] is not None:
        raise ValueError('Research runs must use every valid window.')
    if config['split']['protocol'] != 'grouped_subject_wise_5fold' or config['split']['split_seed'] != 42:
        raise ValueError('The official protocol requires the fixed seed42 five-fold manifest.')
    data, selection = config['data'], config['selection']
    source = data.get('source', 'legacy_wrist')
    allowed_selection = {'val_macro_f1', 'val_subject_macro_f1'} if source == 'common_rate' else {'val_macro_f1'}
    if (config['experiment']['model'] not in ('cnn1d', 'gru') or selection.get('metric') not in allowed_selection
            or selection != {'metric': selection.get('metric'), 'mode': 'max', 'tie_breaker': 'val_loss'}):
        raise ValueError('Unsupported model/selection protocol.')
    if config['experiment']['model'] == 'gru':
        options = config.get('model', {})
        if set(options) - {'hidden_size', 'dropout'}:
            raise ValueError('Unsupported GRU model options.')
        hidden = options.get('hidden_size', 64)
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
            raise ValueError('GRU hidden_size must be a positive integer.')
        if not 0 <= options.get('dropout', 0.3) < 1:
            raise ValueError('GRU dropout must be in [0, 1).')
    elif set(config.get('model', {})) - {'dropout'} or not 0 <= config.get('model', {}).get('dropout', 0.3) < 1:
        raise ValueError('CNN options support only dropout in [0, 1).')
    if config['data']['labels'] != [1, 2, 3] or config['runtime']['parallel_folds'] != 1:
        raise ValueError('Official experiments use labels 1/2/3 and sequential folds.')
    if config['training']['optimizer'] != 'adam' or config['training']['loss'] != 'weighted_cross_entropy':
        raise ValueError('Unsupported optimizer or loss.')
    if not all(config['logging'][key] for key in ('log_every_iteration', 'log_every_epoch', 'flush_each_iteration')):
        raise ValueError('Complete live logging is mandatory.')
    tc = config['training']
    if not tc['seeds'] or len(set(tc['seeds'])) != len(tc['seeds']) or not all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in tc['seeds']):
        raise ValueError('Training seeds must be distinct uint32 integers.')
    if any(tc[key] <= 0 for key in ('max_epochs', 'batch_size', 'learning_rate', 'early_stopping_patience', 'gradient_clip_norm')) or tc['early_stopping_min_delta'] < 0 or tc['weight_decay'] < 0:
        raise ValueError('Invalid training hyperparameters.')
    if source == 'common_rate':
        from src.data.common_rate import CHANNELS
        experiment_folder(config)
        channels = data['channels']
        if (data['sampling_rate'], data['window_seconds'], data['stride_seconds'], data.get('quality_policy')) != (64, 30, 15, 'all_samples_valid'):
            raise ValueError('The common-rate protocol requires 64 Hz, 30/15 s windows and all samples valid.')
        if not channels or channels != [c for c in CHANNELS if c in channels] or not data.get('processed_dir'):
            raise ValueError('Common-rate data require an ordered channel list and processed directory.')
        if config['experiment']['id'] == 'common_rate64_cnn1d_grouped5fold_v1' and (
                tuple(channels) != CHANNELS or config['experiment']['model'] != 'cnn1d'
                or selection['metric'] != 'val_subject_macro_f1'):
            raise ValueError('The official CNN run requires all 14 channels and subject-mean validation selection.')
    elif source == 'legacy_wrist':
        sensors = data['sensors']
        if not sensors or len(set(sensors)) != len(sensors) or not set(sensors) <= {'ACC', 'BVP', 'EDA'}:
            raise ValueError('Unsupported or duplicate sensors.')
    else:
        raise ValueError(f'Unsupported data source: {source}')
    from src.utils.tensorboard_tracker import tracking_options
    tracking_options(config)
    return config


def same_training_config(left, right):
    """Monitoring can change on resume; computation, selection and splits cannot."""
    return {k: v for k, v in left.items() if k not in ('logging', 'tracking')} == {k: v for k, v in right.items() if k not in ('logging', 'tracking')}


def raw_root():
    return Path(os.environ.get('WESAD_ROOT', ROOT / 'data/raw/WESAD')).resolve()
