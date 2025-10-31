#!/usr/bin/env python3
"""Run the InteractiveDataExplorer on the project's creditcard dataset and save plots.

This script sets up the import path so the `train/src` package modules can be imported
and invokes the explorer to load the creditcard.csv and produce visualizations.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
src_path = repo_root / 'train' / 'src'
sys.path.insert(0, str(src_path))

from Data_explore import InteractiveDataExplorer

# creditcard.csv lives under Structured_data_template/app/Data
DATA_PATH = repo_root / 'app' / 'Data' / 'creditcard.csv'

def main():
    explorer = InteractiveDataExplorer()
    ok = explorer.load_data(str(DATA_PATH))
    if not ok:
        print(f"Failed to load data at {DATA_PATH}")
        return

    explorer.quick_overview()
    # Generate profile and visualizations using Class as target
    explorer.generate_comprehensive_profile(target_col='Class')
    explorer.create_visualizations(target_col='Class')
    explorer.analyze_target_relationship('Class')
    explorer.suggest_preprocessing_steps()

    print("Explorer run completed. Plots saved to 'exploration_plots' directory.")

if __name__ == '__main__':
    main()
