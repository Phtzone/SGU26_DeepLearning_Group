import argparse
from src.data.splits import ensure_manifest
from src.utils.config import ROOT
from src.utils.tracking import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    manifest = ensure_manifest(ROOT / f'splits/grouped_5fold_seed{args.seed}.json', args.seed)
    print('Groups:', manifest['groups'])
    print('SHA256:', digest(manifest))
    for fold in manifest['folds']:
        print(fold)


if __name__ == '__main__':
    main()
