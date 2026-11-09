"""Short synthetic event smoke test. Does not train a model or read WESAD."""
import argparse
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.evaluation.metrics import classification_metrics
from src.utils.config import ROOT, load_config
from src.utils.experiment_tracking import TrainingTracker
from src.utils.tracking import identifier, write_json


def smoke(output=None):
    folder = Path(output) if output else ROOT / 'results/tensorboard_upgrade/smoke_events' / identifier()
    if folder.exists():
        raise FileExistsError('Smoke output must be a new directory.')
    config = load_config()
    config['experiment'].update(id='SYNTHETIC_TRACKING_SMOKE', model='synthetic_event_logging')
    config['training']['max_epochs'] = 1
    config['logging']['console']['enabled'] = False
    config['tracking'].update(backend='tensorboard', strict=True)
    config['tracking']['tensorboard']['enabled'] = True
    labels = torch.tensor([0, 0, 0, 1, 2])
    logits = torch.tensor([[2., 0., -1.], [1., 0., -1.], [2., .5, -1.], [1., 0., -.5], [.5, 0., -1.]])
    val_logits = logits.roll(1, dims=1)
    train = classification_metrics(labels.numpy(), logits.argmax(dim=1).numpy())
    val = classification_metrics(labels.numpy(), val_logits.argmax(dim=1).numpy())
    train_loss = torch.nn.functional.cross_entropy(logits, labels).item()
    val_loss = torch.nn.functional.cross_entropy(val_logits, labels).item()
    metadata = {'run_id': identifier(), 'fold_id': 1, 'training_seed': 42, 'manifest_checksum': 'synthetic-not-wesad',
        'effective_config': config, 'device': 'cpu', 'subjects': {role: ['synthetic'] for role in ('train', 'validation', 'test')},
        'window_counts': {role: len(labels) for role in ('train', 'validation', 'test')}, 'trainable_parameters': 0,
        'purpose': 'synthetic_event_smoke_no_training', 'best_epoch': 1, 'completed_epochs': 1,
        'training_duration_seconds': 0, 'checkpoints': {'best': 'N/A: no model training'}, 'test_metrics': val}
    tracker = TrainingTracker(folder / 'logs', config)
    status = 'failed'
    try:
        tracker.start_run(metadata, 'synthetic_smoke', 1)
        for step, loss in enumerate((train_loss, val_loss), 1):
            tracker.log_batch({'run_id': metadata['run_id'], 'attempt_id': 'synthetic', 'fold_id': 1, 'training_seed': 42,
                'epoch': 1, 'batch_index': step - 1, 'global_step': step, 'batch_size': len(labels), 'train_loss': loss,
                'learning_rate': .001, 'gradient_norm': None, 'batch_duration_ms': None, 'timestamp': None})
        row = {'attempt_id': 'synthetic', 'epoch': 1, 'train_online_loss': train_loss, 'train_eval_loss': train_loss,
            'train_accuracy': train['accuracy'], 'train_macro_f1': train['macro_f1'], 'train_balanced_accuracy': train['balanced_accuracy'],
            'val_loss': val_loss, 'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1'], 'val_balanced_accuracy': val['balanced_accuracy'],
            'learning_rate': .001, 'epoch_duration_seconds': None, 'best_epoch': 1, 'best_val_macro_f1': val['macro_f1'], 'early_stopping_counter': 0}
        validation = {'attempt_id': 'synthetic', 'epoch': 1, 'val_loss': val_loss, 'val_accuracy': val['accuracy'],
            'val_macro_f1': val['macro_f1'], 'val_balanced_accuracy': val['balanced_accuracy'],
            'val_precision_per_class': val['precision_per_class'], 'val_recall_per_class': val['recall_per_class'],
            'val_f1_per_class': val['f1_per_class'], 'is_best_checkpoint': True}
        tracker.log_epoch(row, validation, {'bad_epochs': 0, 'is_best': True})
        tracker.log_test(val, metadata, [])
        status = 'completed'
    finally:
        tracker.finish_run(status)
    epochs = EventAccumulator(str(folder / 'tensorboard/epochs'), size_guidance={'scalars': 0}).Reload()
    batches = EventAccumulator(str(folder / 'tensorboard/batches'), size_guidance={'scalars': 0}).Reload()
    images = EventAccumulator(str(folder / 'tensorboard/evaluation')).Reload()
    assert [record.step for record in batches.Scalars('batch/loss')] == [1, 2]
    assert abs(epochs.Scalars('train/accuracy')[0].value - train['accuracy']) < 1e-6
    assert abs(epochs.Scalars('val/accuracy')[0].value - val['accuracy']) < 1e-6
    assert abs(epochs.Scalars('train/eval_loss')[0].value - train_loss) < 1e-6
    assert len(images.Tags()['images']) == 3
    evidence = {'status': 'passed', 'purpose': metadata['purpose'], 'real_data_training_executed': False,
        'output': str(folder), 'epoch_scalar_tags': epochs.Tags()['scalars'], 'batch_steps': [1, 2],
        'evaluation_images': images.Tags()['images'], 'train_accuracy': train['accuracy'], 'val_accuracy': val['accuracy']}
    write_json(folder / 'smoke_verification.json', evidence)
    write_json(ROOT / 'results/tensorboard_upgrade/smoke_verification.json', evidence)
    print(evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    smoke(parser.parse_args().output)


if __name__ == '__main__':
    main()
