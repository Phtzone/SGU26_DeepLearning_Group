import argparse
from src.evaluation.enhanced_visualization import generate_visualizations


def main():
    parser = argparse.ArgumentParser(description='Create new enhanced PNG/SVG figures from existing artifacts. Never trains or overwrites originals.')
    parser.add_argument('--group-id', required=True)
    args = parser.parse_args()
    generate_visualizations(args.group_id)


if __name__ == '__main__':
    main()
