raw_data_filename = "datasets/trenet/clean_gas_data.csv"
trend_filename = 'datasets/trenet/gas_data_Methane(ppm)_pad.pkl'
pc_raw_data_filename = "datasets/trenet/clean_PC_data.csv"
pc_trend_filename = 'datasets/trenet/pc_data_voltage.pkl'

local_data_window = 100
trend_window_size = 3 # was 3, set to 1 for local data
duration = None
train_size = 3974
test_size = 1
shift_train = True
feature_type = 'pointdata'
target = 'trend'

ESTIMATORS = ['SVR', 'RF', 'GBM', 'MLP']
# ESTIMATORS = ['SVR', 'RF', 'GBM', 'MLP', 'LSTM', 'CNN']
HYPER_PARAMETERS = 'hyperparameters/hyperparameters.json'

evaluation = 'recalibration_testing'
metric = 'mse'

parameters = {'read_data': True, 'raw_data_filename': raw_data_filename,
              'trend_filename': trend_filename, 'feature_type': feature_type,
              'main_time_series': 'Methane(ppm)',
              'na_values': None, 'header': 0, 'main_only': True,
              'separator': ',', 'fill_na_method': 'pad',
              'window_size': trend_window_size,
              'local_window': local_data_window,
              'target_name': target,
              # algorithm name and hyper-parameter config file
              'estimator': None, 'param_filename': HYPER_PARAMETERS,
              # evaluation procedure
              'evaluation': evaluation, 'validation': True,
              'new_validation_score': True,
              # 'val_fraction': specs.VALIDATION_FRACTION,
              'train_size': train_size, 'test_size': test_size, 'shift_train': True,
              # 'test_fraction': specs.TEST_FRACTION, 'return_val_score': True,
              'return_train_score': True, 'scoring': metric,
              'verbose': False, 'return_estimator': False
              }

