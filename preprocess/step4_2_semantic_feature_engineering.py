import os
import argparse
import logging

import pandas as pd
import yaml

from utils_for_feature import (
    iter_user_files,
)


def load_config(path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def process_all(config: dict):
    """Process all scenarios and user files to extract sequence features."""
    input_base = config['input_base']
    output_dir = config['output_dir']
    scenarios = config.get('scenarios', [])

    # Iterate through all user files
    for scen, user, filepath in iter_user_files(input_base, scenarios):
        try:
            df = pd.read_csv(filepath)

            # 只保留 http 和 email 行为的必要字段
            filtered = df[df['behavior_sequence'].isin(['http', 'email'])]
            features = filtered[['id', 'date', 'behavior_sequence', 'content', 'label']].copy()

            # 7) Save to output directory
            os.makedirs(output_dir, exist_ok=True)
            out_name = f"{scen}_{user}_semantic.csv"
            out_path = os.path.join(output_dir, out_name)
            features.to_csv(out_path, index=False)
            logging.info(f"Saved features for {scen}/{user} to {out_path}")

        except Exception as e:
            logging.error(f"Failed processing {scen}/{user}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract daily semantic_content features from labeled logs."
    )
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to YAML config file'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    cfg_all = load_config(args.config)
    config = cfg_all.get('semantic_feature_engineering', cfg_all)
    process_all(config)


if __name__ == '__main__':
    main()

