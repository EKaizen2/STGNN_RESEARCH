def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import json
from preprocessing.preproprecessor import to_supervise
from datasets.loader import load_data
#from models.lstm.bohb_lstm_classifier import BOHBLSTMClassifier
from models.bohb_lstm_regressor import BOHBLSTMRegressor
from models.gwn_trend import GWNTrendPredictor
from sklearn.svm import SVC, SVR
from lightgbm import LGBMClassifier, LGBMRegressor
from evaluation.validation import time_series_validation, \
                                            time_series_testing, \
                                            one_holdout_validation, \
                                            one_holdout_testing, \
                                            validaton_with_recalibration, \
                                            testing_with_recalibration
from numpy import unique as np_unique
from copy import deepcopy


def run(X=None, Y=None, read_data=True, raw_data=None, trends=None, raw_data_filename=None,
        trend_filename=None, main_time_series='Close', na_values=None, header=0,
        main_only=True, separator=',', fill_na_method='pad',
        # preprocessing
        last_n_days=False, strength_metric='angle', angle_metric='degree',
        duration=None, return_segment=False, overlap=False, overlap_fraction=0.0,
        pla_algorithm='bottom', error=False, max_error=0.0, n_segments=4419,
        feature_type='trend', window_size=1, target_name='trend', n_classes=3, line_fitter='regression',
        lower=-0.5, upper=0.55, local_window=100, lag=30, n_trend=True, max_feature_size=None,
        # algorithm name and hyper-parameter config file
        estimator='rf_regressor', param_filename="../hyperparameters/hyperparameters",
        # evaluation procedure
        evaluation='testing', n_splits=5, validation=True, val_fraction=0.15, test_fraction=0.15, return_val_score=True,
        return_train_score=False, scoring='neg_mean_squared_error', verbose=1, return_estimator=False, n_jobs=1,
        train_size=None, test_size=None, shift_train=True,
        new_validation_score=True, y_transformer=None,
        # automl config
        config=None):

    if X is None and Y is None:
        if read_data:
            try:
                trends = load_data(trend_filename, is_pickle=True)
            except ValueError:
                pass
            if last_n_days or feature_type.lower() in ['pointdata', 'trend_local_data', 'strength_local_data']:
                raw_data = load_data(raw_data_filename, is_pickle=False,
                                     main_time_series=main_time_series, na_values=na_values,
                                     header=header, main_only=main_only, separator=separator,
                                     fill_na_method=fill_na_method)

        (X, Y), config = live_object_wrapper(live_object=to_supervise, config=config, raw_data=raw_data, trends=trends,
                                             last_n_days=last_n_days, strength_metric=strength_metric,
                                             angle_metric=angle_metric, duration=duration, return_segment=return_segment,
                                             overlap=overlap, overlap_fraction=overlap_fraction, line_fitter=line_fitter,
                                             pla_algorithm=pla_algorithm, error=error, max_error=max_error,
                                             n_segments=n_segments, feature_type=feature_type, window_size=window_size,
                                             target_name=target_name, n_classes=n_classes, lower=lower, upper=upper,
                                             local_window=local_window, lag=lag, max_feature_size=max_feature_size,
                                             n_trend=n_trend)
        # print("Shape of X: ", X.shape)
    if config is not None:
        estimator = config['algorithm']
    estimator, config = live_object_wrapper(live_object=get_estimator, estimator=estimator,
                                            param_filename=param_filename, config=config,
                                            input_dim=X.shape[-1], output_dim=get_output_dim(estimator, Y),
                                            trend_dim=X.shape[-1]-local_window, local_data_size=local_window)

    results, _ = live_object_wrapper(live_object=evaluate, config=config, estimator=estimator, X=X, Y=Y,
                                     evaluation=evaluation, n_splits=n_splits, validation=validation,
                                     val_fraction=val_fraction, test_fraction=test_fraction, train_size=train_size,
                                     test_size=test_size, shift_train=shift_train,
                                     new_validation_score=new_validation_score,
                                     return_train_score=return_train_score, scoring=scoring, verbose=verbose,
                                     return_val_score=return_val_score, return_estimator=return_estimator,
                                     n_jobs=n_jobs, y_transformer=y_transformer)
    return results


def get_output_dim(estimator_name, Y):
    estimator_name = estimator_name.lower()
    if 'regressor' in estimator_name or estimator_name in ['svr', 'trenet']:
        return Y.shape[-1]
    elif 'classifier' in estimator_name or 'svm' in estimator_name:
        return len(np_unique(Y))


def to_supervise_wrapper(config=None, **kwargs):
    if config is None:
        return to_supervise(**kwargs), config
    to_supervise_args = to_supervise.__code__.co_varnames[:to_supervise.__code__.co_argcount]
    _config = deepcopy(config)
    for param in config:
        if param in to_supervise_args:
            kwargs[param] = config[param]
            del _config[param]
    return to_supervise(**kwargs), _config


def live_object_wrapper(live_object, config, **kwargs):
    if config is None:
        return live_object(**kwargs), config
    kwargs, _config = replace_default_params(live_object=live_object, config=config, **kwargs)
    kwargs['config'] = deepcopy(_config)
    return live_object(**kwargs), _config


def replace_default_params(live_object, config, **kwargs):
    live_object_args = live_object.__code__.co_varnames[:live_object.__code__.co_argcount]
    _config = deepcopy(config)
    for param in config:
        if param in live_object_args:
            kwargs[param] = config[param]
            del _config[param]
    return kwargs, _config


def evaluate_wrapper(config=None, **kwargs):
    if config is None:
        return evaluate(**kwargs), config
    evaluate_args = evaluate.__code__.co_varnames[:evaluate.__code__.co_argcount]
    _config = deepcopy(config)
    for param in config:
        if param in evaluate_args:
            kwargs[param] = config[param]
            del _config[param]
    return evaluate(**kwargs), _config


def evaluate(estimator, X, Y, evaluation='testing', n_splits=5, validation=True,
             test_fraction=0.15, val_fraction=0.1, return_train_score=True, scoring='neg_mean_squared_error',
             train_size=None, test_size=None, shift_train=True, new_validation_score=True,
             verbose=1, return_val_score=True, return_estimator=False, n_jobs=1, y_transformer=None, config=None):

    evaluation = evaluation.lower()
    if evaluation == 'recalibration_testing':
        return testing_with_recalibration(estimator, X, Y, train_size=train_size, test_size=test_size,
                                          shift_train=shift_train, scoring=scoring,
                                          new_validation_score=new_validation_score,
                                          return_train_score=return_train_score, verbose=verbose,
                                          n_jobs=n_jobs, y_transformer=y_transformer)
    if evaluation == 'recalibration_validation':
        return validaton_with_recalibration(estimator, X, Y, train_size=train_size, test_size=test_size,
                                            shift_train=shift_train, scoring=scoring,
                                            return_train_score=return_train_score,
                                            new_validation_score=new_validation_score,
                                            return_estimator=return_estimator, verbose=verbose, n_jobs=n_jobs)
    elif evaluation == 'testing':
        return time_series_testing(estimator, X, Y, n_splits=n_splits, val_fraction=val_fraction,
                                   return_train_score=return_train_score, scoring=scoring,
                                   verbose=verbose, n_jobs=n_jobs)
    elif evaluation == 'validation':
        return time_series_validation(estimator, X, Y, n_splits=n_splits, val_fraction=val_fraction,
                                      return_train_score=return_train_score, scoring=scoring,
                                      verbose=verbose, return_estimator=return_estimator, n_jobs=n_jobs)
    elif evaluation == 'traditional_validation':
        return one_holdout_validation(estimator, X, Y, validation=validation, test_fraction=test_fraction,
                                      val_fraction=val_fraction,  scoring=scoring,
                                      return_train_score=return_train_score, return_estimator=return_estimator)
    elif evaluation == 'traditional_testing':
        return one_holdout_testing(estimator, X, Y, validation=validation, test_fraction=test_fraction,
                                   val_fraction=val_fraction,  scoring=scoring, return_train_score=return_train_score,
                                   return_val_score=return_val_score, return_estimator=return_estimator)
    else:
        print("Evaluation should be 'testing', 'validation' or 'traditional'.")
        exit(-1)


def get_estimator(estimator, param_filename="hyperparameters/hyperparameters.json",
                  config=None, input_dim=None, output_dim=None,
                  trend_dim=None, local_data_size=None):
    estimator = estimator.lower()
    with open(param_filename) as f:
        params = json.load(f)[estimator]
    if config is not None:
        params = normalise_parameters(params, config)
    if 'lstm' in estimator:
        params['input_dim'] = input_dim
        params['output_dim'] = output_dim
        estimator = _get_estimator(estimator)(**params)
    else:
        estimator = _get_estimator(estimator)(**params)
    return estimator


def normalise_parameters(default_params, config):
    _config = deepcopy(config)
    del _config['algorithm']
    for param in _config:
        default_params[param] = _config[param]
    return default_params


def _get_estimator(estimator):
    estimator = estimator.lower()
    if estimator == 'bohb_lstm_regressor':
        return BOHBLSTMRegressor
    if estimator == 'gwn':
        return GWNTrendPredictor
    else:
        print(estimator + " not implemented")
        exit(-1)

