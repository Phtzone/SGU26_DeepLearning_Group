from src.data.inspection import inspect_dataset
from src.utils.config import ROOT, raw_root


def main():
    report = inspect_dataset(raw_root(), ROOT / 'data/interim/inspection.json')
    print(f"Inspected {len(report['subjects'])}/15 subjects: {report['status']}")


if __name__ == '__main__':
    main()
