import argparse
import re
from src.utils.config import ROOT
from src.evaluation.aggregation import aggregate_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.group_id):
        raise ValueError('Invalid group ID.')
    summary = aggregate_group(ROOT / 'runs/experiment_e1' / args.group_id)
    print(summary)


if __name__ == '__main__':
    main()
