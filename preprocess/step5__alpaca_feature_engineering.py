# -*- coding: utf-8 -*-

import os
import argparse
import logging
import json

import pandas as pd
import yaml

from utils_for_feature import (
    iter_user_files,
)


def load_config(path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def convert_behavior_sequence_to_alpaca_style(df: pd.DataFrame, output_path: str):
    alpaca_data = []

    for _, row in df.iterrows():
        behavior_text = str(row['compressed_seq']).strip()
        label = int(row['label'])
        score = 1.0 if label == 1 else 0.0
        prediction_str = "Abnormal" if label == 1 else "Normal"

        alpaca_data.append({
            "instruction": "Please analyze the following behavior sequence. Respond with both an anomaly score and a classification result (‘Normal‘ or ‘Abnormal‘)",
            "input": behavior_text,
            "output": f"Anomaly Score = {score:.2f}, Prediction = “{prediction_str}”"
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(alpaca_data)} Alpaca behavior records to {output_path}")


def convert_semantic_content_to_alpaca_style(df: pd.DataFrame, output_path: str):
    alpaca_data = []

    for _, row in df.iterrows():
        behavior_text = str(row['content']).strip()
        label = int(row['label'])
        score = 1.0 if label == 1 else 0.0
        prediction_str = "Abnormal" if label == 1 else "Normal"

        alpaca_data.append({
            "instruction": "Please analyze the following behavior sequence. Respond with both an anomaly score and a classification result (‘Normal‘ or ‘Abnormal‘)",
            "input": behavior_text,
            "output": f"Anomaly Score = {score:.2f}, Prediction = “{prediction_str}”"
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(alpaca_data)} Alpaca semantic records to {output_path}")


def process_all(config: dict):
    """Convert behavior and semantic logs to Alpaca format based on file names."""
    input_behavior_sequence_dir = config['input_behavior_sequence']
    input_semantic_content_dir = config['input_semantic_content']
    output_dir = config['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_behavior_sequence_dir):
        if not fname.endswith("_behavior.csv"):
            continue

        try:
            parts = fname.replace(".csv", "").split("_")
            if len(parts) < 3:
                logging.warning(f"Invalid filename format: {fname}")
                continue

            scenario = parts[0] + "_" + parts[1]  # e.g., scenario_2
            user_id = parts[2]                   # e.g., AAF0535

            behavior_path = os.path.join(input_behavior_sequence_dir, fname)
            semantic_fname = f"{scenario}_{user_id}_semantic.csv"
            semantic_path = os.path.join(input_semantic_content_dir, semantic_fname)

            if not os.path.exists(semantic_path):
                logging.warning(f"Missing semantic file for {fname}, skipping...")
                continue

            df_behavior = pd.read_csv(behavior_path)
            df_semantic = pd.read_csv(semantic_path)

            out_behavior = os.path.join(output_dir, f"{scenario}_{user_id}_alpaca_behavior.json")
            out_semantic = os.path.join(output_dir, f"{scenario}_{user_id}_alpaca_semantic.json")

            convert_behavior_sequence_to_alpaca_style(df_behavior, out_behavior)
            convert_semantic_content_to_alpaca_style(df_semantic, out_semantic)

            logging.info(f"✅ Processed {scenario}/{user_id}")

        except Exception as e:
            logging.error(f"❌ Failed to process {fname}: {e}")




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
    config = cfg_all.get('alpaca_feature_engineering', cfg_all)
    process_all(config)


if __name__ == '__main__':
    main()

