import os
import pandas as pd
import numpy as np
from datasets.loader import load_data
from preprocessing.features import trend

def prepare_data(dataset_name, **kwargs):
    """
    Prepare and save both trend segments and raw data for all datasets
    Args:
        dataset_name: Name of the dataset to process
        **kwargs: Additional parameters for preprocessing
    """
    # Create processed_data directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    
    # Load raw data
    if dataset_name == 'voltage':
        raw_data = load_data("datasets/clean_PC_data.csv", 
                            is_pickle=False, 
                            main_time_series='Voltage',
                            na_values=None, 
                            header=0, 
                            main_only=False,  # Changed to False to get all columns
                            separator=',', 
                            fill_na_method='pad')
    elif dataset_name == 'methane':
        raw_data = load_data("datasets/clean_gas_data.csv",
                            is_pickle=False,
                            main_time_series='Methane(ppm)',
                            na_values=None,
                            header=0,
                            main_only=False,  # Changed to False to get all columns
                            separator=',',
                            fill_na_method='pad')
    elif dataset_name == 'nyse':
        raw_data = load_data("datasets/NYSE.csv",
                            is_pickle=False,
                            main_time_series='Close',
                            na_values=None,
                            header=0,
                            main_only=False,  # Changed to False to get all columns
                            separator=',',
                            fill_na_method='pad')
    elif dataset_name == 'jse':
        raw_data = load_data("datasets/JSE.csv",
                            is_pickle=False,
                            main_time_series='ASPEN',
                            na_values=None,
                            header=0,
                            main_only=False,  # Changed to False to get all columns
                            separator=',',
                            fill_na_method='pad')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Process and save trend segments for each column
    trends_dict = {}
    for column in raw_data.columns:
        trends = trend(raw_data[column], strength_metric='angle', angle_metric='degree',
                      duration=None, return_segment=False, overlap=False,
                      overlap_fraction=False, pla_algorithm='bottom', slope_estimator='regression',
                      error=False, max_error=False, n_segments=10015)
        trends_dict[column] = trends
    
    # Save trend segments
    trend_path = f'processed_data/{dataset_name}_trends.pkl'
    pd.to_pickle(trends_dict, trend_path)
    print(f"Trend segments saved to {trend_path}")
    
    # Save raw data
    raw_path = f'processed_data/{dataset_name}_raw.pkl'
    pd.to_pickle(raw_data, raw_path)
    print(f"Raw data saved to {raw_path}")
    
    return trend_path, raw_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for trend prediction models')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (voltage, methane, nyse, jse)')
    
    args = parser.parse_args()
    
    prepare_data(args.dataset) 