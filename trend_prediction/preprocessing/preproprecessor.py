from preprocessing.features import trend, local_data, movement_direction, movement_from_last_n
from preprocessing.util import targets, sliding_window
from numpy import concatenate as np_concat
from evaluation.validation import rate_of_change_to_pointdata
import pandas as pd
import numpy as np


def model_specific_preprocessing(raw_data, model_type='lstm', feature_type='trend', **kwargs):
    """
    Preprocess data based on model type and feature type
    Args:
        raw_data: Input data
        model_type: Type of model ('lstm' or 'gwn')
        feature_type: Type of features to use ('trend' or 'pointdata')
        **kwargs: Additional parameters for preprocessing
    Returns:
        Processed data and metadata
    """
    if model_type.lower() == 'lstm':
        # For LSTM, we use univariate processing
        if feature_type == 'trend':
            # Process trend segments for single column
            trends = trend(raw_data, strength_metric='angle', angle_metric='degree',
                          duration=None, return_segment=False, overlap=False,
                          overlap_fraction=False, pla_algorithm='bottom', slope_estimator='regression',
                          error=False, max_error=False, n_segments=10015)
            
            # Extract trend features (strength and duration)
            processed_data = trends[['strength', 'duration']].values
            
            metadata = {
                'num_nodes': 1,  # Single node for LSTM
                'num_features': 2,  # strength and duration
                'num_timesteps': len(processed_data),
                'feature_type': 'trend'
            }
        else:  # pointdata
            # For LSTM, ensure we have a single column
            if isinstance(raw_data, pd.DataFrame):
                if raw_data.shape[1] > 1:
                    raise ValueError("LSTM model requires univariate data (single column)")
                processed_data = raw_data.values
            else:
                processed_data = raw_data.reshape(-1, 1) if len(raw_data.shape) == 1 else raw_data
                
            metadata = {
                'num_nodes': 1,  # Single node for LSTM
                'num_features': 1,  # One feature per node
                'num_timesteps': len(processed_data),
                'feature_type': 'pointdata'
            }
            
    elif model_type.lower() == 'gwn':
        # For GWN, we process based on feature type
        if feature_type == 'trend':
            # Process trend segments for each column
            if isinstance(raw_data, pd.DataFrame):
                all_trends = []
                for column in raw_data.columns:
                    column_data = raw_data[column].values
                    trends = trend(column_data, strength_metric='angle', angle_metric='degree',
                                 duration=None, return_segment=False, overlap=False,
                                 overlap_fraction=False, pla_algorithm='bottom', slope_estimator='regression',
                                 error=False, max_error=False, n_segments=10015)
                    # Each stock's trends are [strength, duration]
                    all_trends.append(trends[['strength', 'duration']].values)
                
                # Stack trends for each node
                # Shape: [num_timesteps, num_nodes, num_features]
                processed_data = np.stack(all_trends, axis=1)
                
                # Apply sliding window of size 21 for prediction
                window_size = 21
                processed_data = sliding_window(processed_data, window_size)
                # Reshape to [num_samples, num_nodes, num_features, window_size]
                processed_data = processed_data.reshape(-1, processed_data.shape[1], 2, window_size)
                # Transpose to [num_samples, num_nodes, window_size, num_features]
                processed_data = np.transpose(processed_data, (0, 1, 3, 2))
            else:
                # If single column, process as is
                trends = trend(raw_data, strength_metric='angle', angle_metric='degree',
                             duration=None, return_segment=False, overlap=False,
                             overlap_fraction=False, pla_algorithm='bottom', slope_estimator='regression',
                             error=False, max_error=False, n_segments=10015)
                # Shape: [num_timesteps, 1, num_features]
                processed_data = trends[['strength', 'duration']].values.reshape(-1, 1, 2)
                
                # Apply sliding window of size 21 for prediction
                window_size = 21
                processed_data = sliding_window(processed_data, window_size)
                # Reshape to [num_samples, 1, num_features, window_size]
                processed_data = processed_data.reshape(-1, 1, 2, window_size)
                # Transpose to [num_samples, 1, window_size, num_features]
                processed_data = np.transpose(processed_data, (0, 1, 3, 2))
            
            metadata = {
                'num_nodes': processed_data.shape[1],  # Number of nodes (stocks)
                'num_features': 2,  # strength and duration per node
                'num_timesteps': 21,  # Fixed window size for prediction
                'feature_type': 'trend'
            }
            
        elif feature_type == 'pointdata':
            # Process raw point data
            if isinstance(raw_data, pd.DataFrame):
                processed_data = raw_data.select_dtypes(include=[np.number])
            else:
                processed_data = pd.DataFrame(raw_data)
            
            # For GWN, we need to process each node (stock) separately
            all_windows = []
            for column in processed_data.columns:
                # Get data for this node
                node_data = processed_data[column].values
                
                # Apply sliding window of size 21 for prediction
                window_size = 21
                windows = sliding_window(node_data, window_size)
                
                # Reshape to [num_samples, 1, window_size]
                windows = windows.reshape(-1, 1, window_size)
                all_windows.append(windows)
            
            # Stack all nodes together
            # Shape: [num_samples, num_nodes, window_size]
            processed_data = np.stack(all_windows, axis=1)
            
            # Add feature dimension
            # Shape: [num_samples, num_nodes, window_size, 1]
            processed_data = processed_data.reshape(*processed_data.shape, 1)
            
            metadata = {
                'num_nodes': processed_data.shape[1],  # Number of nodes (stocks)
                'num_features': 1,  # One feature per node (closing price)
                'num_timesteps': 21,  # Fixed window size for prediction
                'feature_type': 'pointdata'
            }
        else:
            raise ValueError(f"Unsupported feature type for GWN: {feature_type}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
            
    return processed_data, metadata


def to_supervise(raw_data=None, trends=None, last_n_days=False, strength_metric='angle',
                 angle_metric='degree', duration=None, return_segment=False, overlap=False, overlap_fraction=0.0,
                 pla_algorithm='bottom', error=False, max_error=0.0, n_segments=4419, feature_type='trend',
                 window_size=1, target_name='trend', n_classes=3, lower=-0.5, upper=0.55, local_window=100,
                 lag=30, max_feature_size=None, n_trend=True, line_fitter='regression', config=None):
    if last_n_days:
        return movement_from_last_n(raw_data, lag=lag, duration=duration, overlap=overlap,
                                    overlap_fraction=overlap_fraction, max_feature_size=max_feature_size,
                                    n_trend=n_trend, n_segments=n_segments, angle_metric=angle_metric)

    if trends is None:
        trends = trend(raw_data, strength_metric=strength_metric, angle_metric=angle_metric,
                       duration=duration, return_segment=return_segment, overlap=overlap,
                       overlap_fraction=overlap_fraction, pla_algorithm=pla_algorithm, error=error,
                       max_error=max_error, n_segments=n_segments, slope_estimator=line_fitter)
    directions = None
    if feature_type == 'trend':
        features = trends[['strength', 'duration']].values
        X = sliding_window(features, window_size)
    elif feature_type == 'strength':
        features = trends[['strength']].values
        X = sliding_window(features, window_size)
    elif feature_type == 'direction':
        directions = movement_direction(trends['strength'], n_classes=n_classes,
                                        lower=lower, upper=upper)
        X = sliding_window(directions, window_size)
    elif feature_type == 'pointdata':
        X = local_data(raw_data, trends['index'].values, trend_window=window_size, local_window=local_window)
    elif feature_type == 'trend_local_data':
        _trend = sliding_window(trends[['strength', 'duration']].values, window_size)
        _local_data = local_data(raw_data, trends['index'].values, trend_window=window_size, local_window=local_window)
        X = np_concat((_trend, _local_data), axis=1)
    elif feature_type == 'strength_local_data':
        _trend = sliding_window(trends[['strength']].values, window_size)
        _local_data = local_data(raw_data, trends['index'].values, trend_window=window_size, local_window=local_window)
        X = np_concat((_trend, _local_data), axis=1)
    else:
        X = sliding_window(trends[feature_type].values, window_size)

    if target_name == 'trend':
        Y = targets(trends[['strength', 'duration']], window=window_size)
    elif target_name == 'strength':
        Y = targets(trends[['strength']], window=window_size)
    elif target_name == 'duration':
        Y = targets(trends[['duration']], window=window_size)
    elif target_name == 'direction':
        if directions is None:
            directions = movement_direction(trends['strength'], n_classes=n_classes,
                                            lower=lower, upper=upper)
        Y = targets(directions, window=window_size)
    elif target_name == 'pointdata':
        Y = targets(trends[['strength']], window=window_size)
        Y = rate_of_change_to_pointdata(X, Y, trends['index'][1] + 1)
    else:
        # if target_name = 'strength' or 'duration'
        Y = targets(trends.loc[:, [target_name]], window=window_size)
    # print((X.shape, Y.shape))
    # import sys
    # sys.exit()
    # import pandas as pd
    # from matplotlib import pyplot as plt
    # print(pd.DataFrame(Y).describe())
    # print(pd.DataFrame(Y).head())
    # print(pd.DataFrame(Y).plot())
    # plt.show()
    return X, Y

if __name__ == '__main__':
    from trend_prediction.datasets.loader import load_data
    import pandas as pd
    from matplotlib import pyplot as plt
    import sys

    dataset = "jse"
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()

    if dataset == "nyse":
        data = load_data("datasets/NYSE.csv", is_pickle=False, main_time_series='Close',
                         na_values=None, header=0, main_only=True, separator=',', fill_na_method='pad')
        trends = trend(data=data, strength_metric='angle', angle_metric='degree',
                       duration=None, return_segment=False, overlap=False,
                       overlap_fraction=False, pla_algorithm='bottom', slope_estimator='regression',
                       error=False, max_error=False, n_segments=10015)
        data = pd.DataFrame({"Closing Price": data})
        x_label = "Daily time step"
        y_label = "Price (USD)"

        trends.to_pickle('processed_data/nyse.pkl')
        print(trends.describe())
        print(trends.head())
    elif dataset == "jse":
        data = load_data("datasets/JSE.csv", is_pickle=False, main_time_series='ASPEN',
                         na_values=None, header=0, main_only=True, separator=',', fill_na_method='pad')
        trends = trend(data=data, strength_metric='angle', angle_metric='degree',
                       duration=None, return_segment=False, overlap=False,
                       overlap_fraction=False, pla_algorithm='bottom', slope_estimator='regression',
                       error=False, max_error=False, n_segments=1001)
        data = pd.DataFrame({"Closing Price": data})
        x_label = "Daily time step"
        y_label = "Price (USD)"

        trends.to_pickle('processed_data/jse.pkl')
        print(trends.describe())
        print(trends.head())
        print(trends.tail())
    print(data.describe())
    plt.rcParams.update({'font.size': 14})
    axes = data.plot()
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    plt.show()
