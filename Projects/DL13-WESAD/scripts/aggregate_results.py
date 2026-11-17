import argparse
import re
from src.utils.config import ROOT
from src.evaluation.aggregation import aggregate_group
from src.evaluation.history import resolve_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.group_id):
        raise ValueError('Invalid group ID.')
    summary = aggregate_group(resolve_group(args.group_id), update_docs=False)
    print(summary)


if __name__ == '__main__':
    main()
