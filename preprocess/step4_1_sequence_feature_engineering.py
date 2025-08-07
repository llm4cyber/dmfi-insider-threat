# sequence_feature_engineering.py
# !/usr/bin/env python3

import os
import argparse
import logging
import pandas as pd
import yaml

from utils_for_feature import (
    determine_working_hours,
    add_work_status,
    process_pc_user_data,
    determine_action,
    determine_time,
    safe_encode_action,
    aggregate_daily_sequences,
    iter_user_files,
    load_mappings,
    join_behavior_sequence,
    compress_behavior_sequence
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

    # Load reference data
    ldap_df = pd.read_csv(config['ldap_path'])
    http_df = pd.read_csv(config['http_csv'])

    # Load mapping dictionaries
    behavior_map, device_map, time_map = load_mappings(config)

    # Iterate through all user files
    for scen, user, filepath in iter_user_files(input_base, scenarios):
        try:
            df = pd.read_csv(filepath)

            # 1) Determine working hours and add status
            start_hr, end_hr = determine_working_hours(df)
            df = add_work_status(df, start_hr, end_hr)

            # 2) Determine PC user type
            df = process_pc_user_data(df, ldap_df)

            # 3) Determine action labels and filter
            df[['action', 'object_']] = df.apply(
                lambda r: pd.Series(determine_action(r, ldap_df, http_df)),
                axis=1
            )
            df = df[df['action'].notna()]

            df['action_seq'] = df.apply(
                lambda r: safe_encode_action(r),
                axis=1
            )

            df = df[df['action_seq'].notna()]

            # 5) Compute time differences and date_only
            df = determine_time(df)

            # 6) Aggregate into daily sequences
            features = aggregate_daily_sequences(df)

            features['compressed_seq'] = features['action_seq'].apply(compress_behavior_sequence)
            features['raw_seq'] = features['action_seq'].apply(join_behavior_sequence)

            # 7) Save to output directory
            os.makedirs(output_dir, exist_ok=True)
            out_name = f"{scen}_{user}_behavior.csv"
            out_path = os.path.join(output_dir, out_name)
            features.to_csv(out_path, index=False)
            logging.info(f"Saved features for {scen}/{user} to {out_path}")

        except Exception as e:
            logging.error(f"Failed processing {scen}/{user}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract daily sequence features from labeled logs."
    )
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to YAML config file'
    )
    args = parser.parse_args()

    print("11111")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    print("22222")
    cfg_all = load_config(args.config)
    config = cfg_all.get('sequence_feature_engineering', cfg_all)
    process_all(config)


if __name__ == '__main__':
    main()





