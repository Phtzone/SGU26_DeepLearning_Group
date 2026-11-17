import argparse
from src.data.preprocessing import prepare_dataset
from src.utils.config import ROOT, raw_root, load_config
from src.utils.tracking import digest


def cache_path(config):
    return ROOT / 'data/processed' / digest(config['data'])[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='experiments/e1_cnn/config.yaml')
    args = parser.parse_args()
    config = load_config(args.experiment)
    metadata = prepare_dataset(raw_root(), cache_path(config), config)
    print('Cache:', cache_path(config))
    print('Windows:', sum(m['shape'][0] for m in metadata))


if __name__ == '__main__':
    main()
