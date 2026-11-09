import argparse
import json
from src.utils.history_import import import_history


def main():
    parser = argparse.ArgumentParser(description='Replay existing committed metrics. No training or raw/checkpoint uploads.')
    parser.add_argument('--group-id', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--mode', choices=['offline', 'online'], default='offline')
    parser.add_argument('--project')
    parser.add_argument('--entity')
    args = parser.parse_args()
    overrides = {key: value for key, value in {'project': args.project, 'entity': args.entity}.items() if value is not None}
    print(json.dumps(import_history(args.group_id, args.mode, args.dry_run,
                     overrides={'tracking': {'wandb': overrides}}), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
