from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from metrics.classification import accuracy, report, combined_report_loss,\
    mcc, confusion_matrix, f1, custom_metric
from metrics.regression import rmse, mse, r2, mae
from numpy import abs as np_abs
import numpy as np
from evaluation.partition import TimeSeriesPartition


def rate_of_change_to_pointdata(x, y, duration):
    return (duration - 1) * np.tan(y * np.pi / 180.0) + x[:, -1].reshape(y.shape)


def score(scoring='mse'):
    scoring = scoring.lower()
    if scoring == 'mse':
        return mse
    elif scoring == 'rmse':
        return rmse
    elif scoring == 'r2':
        return r2
    elif scoring == 'accuracy':
        return accuracy
    elif scoring == 'report':
        return report
    elif scoring == 'report_loss':
        return combined_report_loss
    elif scoring == 'f1':
        return f1
    elif scoring == 'mcc':
        return mcc
    elif scoring == 'confusion_matrix':
        return confusion_matrix
    else:
        print('Scoring ', scoring, "not implemented.")
        exit(-1)


def validaton_with_recalibration(estimator, X, Y, train_size, test_size, shift_train=True,
                                 return_train_score=True, scoring=None, verbose=1,
                                 return_estimator=False, new_validation_score=True,
                                 new_score='mse', n_jobs=-1):

    if train_size < 1:
        data_size = Y.shape[0]
        temp_train_size = train_size
        train_size = int(train_size * data_size)
        if int(temp_train_size + test_size * 2) == 1 or test_size < 1:
            test_size = (data_size - train_size) // 2
        del temp_train_size

    partitions = TimeSeriesPartition(Y, train_size, test_size, shift_train=shift_train, validation=True).split()
    train_validation = list(_train_validation(partitions))
    results = cross_validate(estimator, X, Y, scoring=scoring, cv=train_validation,
                             return_train_score=return_train_score, verbose=verbose,
                             return_estimator=return_estimator, n_jobs=n_jobs)
    if new_validation_score:
        estimators = results['estimator']
        scorer = score(new_score)
        y_true, y_pred = [], []
        for (_, test), estimator_ in zip(train_validation, estimators):
            y_true.append(Y[test])
            y_pred.append(estimator_.predict(X[test]))
        results['test_score'] = scorer(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1))
    return results


def testing_with_recalibration(estimator, X, Y, train_size, test_size, shift_train=True,
                               return_train_score=True, scoring='neg_mean_squared_error',
                               new_validation_score=True, verbose=1, n_jobs=-1, y_transformer=None):
    if train_size < 1:
        data_size = Y.shape[0]
        temp_train_size = train_size
        train_size = int(train_size * data_size)
        if int(temp_train_size + test_size * 2) == 1 or test_size < 1:
            test_size = (data_size - train_size) // 2
        del temp_train_size
    # train_val_results = validaton_with_recalibration(estimator, X, Y, train_size,
    #                                                  test_size, shift_train=shift_train,
    #                                                  return_train_score=return_train_score,
    #                                                  scoring=custom_metric(),
    #                                                  new_validation_score=new_validation_score,
    #                                                  verbose=verbose, return_estimator=True, n_jobs=n_jobs)
    train_val_results = validaton_with_recalibration(estimator, X, Y, train_size,
                                                     test_size, shift_train=shift_train,
                                                     return_train_score=return_train_score,
                                                     scoring='neg_mean_squared_error',
                                                     new_validation_score=new_validation_score,
                                                     verbose=verbose, return_estimator=True, n_jobs=n_jobs)
    partitions = TimeSeriesPartition(Y, train_size, test_size, shift_train=shift_train, validation=True)
    estimators = train_val_results['estimator']
    scorer = score(scoring)
    y_true, y_pred = [], []
    for (_, val, test), estimator_ in zip(partitions.split(), estimators):
        if y_transformer is not None:
            y_true.append(y_transformer(X[test], Y[test]))
            y_pred.append(y_transformer(X[test], estimator_.predict(X[test])))
        else:
            y_true.append(Y[test])
            y_pred.append(estimator_.predict(X[test]))
    return_ = {'test_score': scorer(np.array(y_true).reshape(-1, Y.shape[-1]),
                                    np.array(y_pred).reshape(-1, Y.shape[-1]))}
    # return_ = {'test_score': scorer(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1))}
    if return_train_score:
        return_['train_val_info'] = train_val_results
    return return_


def time_series_validation(estimator, X, Y, n_splits=5, val_fraction=0.1, return_train_score=True,
                           scoring='mse', verbose=1, return_estimator=False, n_jobs=1):
    """Validate the model using time series cross validation"""
    train_validation = _train_validation(partition(Y, n_splits=n_splits, val_fraction=val_fraction))
    return cross_validate(estimator, X, Y, scoring=scoring,
                          cv=train_validation, return_train_score=return_train_score, verbose=verbose,
                          return_estimator=return_estimator, n_jobs=n_jobs)


def one_holdout_validation(estimator, X, Y, test_fraction=0.1, validation=False, val_fraction=0.1,
                           scoring='mse', return_train_score=True, return_estimator=False):
    train_val_end = int((1 - test_fraction) * len(X))
    # test = range(train_val_end, len(X))
    if validation:
        val_end = train_val_end
        train_val_fraction = 1 - test_fraction
        train_fraction = (train_val_fraction - val_fraction) / train_val_fraction
        train_end = int(train_fraction * train_val_end)
    else:
        train_end = train_val_end
    train = range(train_end)
    val = range(train_end, val_end)
    model = estimator.fit(X[train], Y[train])
    # train_score = mse(Y[train], model.predict(X[train]), 'raw_values')
    # test_score = mse(Y[test], model.predict(X[test]), 'raw_values')
    scorer = score(scoring)
    train_score = scorer(Y[train], model.predict(X[train]))
    test_score = scorer(Y[val], model.predict(X[val]))
    return_ = {'test_score': test_score}
    if return_train_score:
        return_['train_score'] = train_score
    if return_estimator:
        return_['estimator'] = model
    return return_


def one_holdout_testing(estimator, X, Y, test_fraction=0.1, validation=False, val_fraction=0.1,
                        scoring='mse', return_train_score=True, return_val_score=True, return_estimator=False):
    train_val_end = int((1 - test_fraction) * len(X))
    test = range(train_val_end, len(X))
    if validation:
        val_end = train_val_end
        train_val_fraction = 1 - test_fraction
        train_fraction = (train_val_fraction - val_fraction) / train_val_fraction
        train_end = int(train_fraction * train_val_end)
        val = range(train_end, val_end)
    else:
        train_end = train_val_end
    train = range(train_end)
    model = estimator.fit(X[train], Y[train])
    scorer = score(scoring)
    train_score = scorer(Y[train], model.predict(X[train]))
    if validation:
        val_score = scorer(Y[val], model.predict(X[val]))
    test_score = scorer(Y[test], model.predict(X[test]))
    return_ = {'test_score': test_score}
    if return_train_score:
        return_['train_score'] = train_score
    if validation and return_val_score:
        return_['validation_score'] = val_score
    if return_estimator:
        return_['estimator'] = model
    return return_


def time_series_testing(estimator, X, Y, n_splits=5, val_fraction=0.1,
                        return_train_score=True, scoring='neg_mean_squared_error',
                        verbose=1, n_jobs=-1):
    train_val_results = time_series_validation(estimator, X, Y,
                                               n_splits=n_splits,
                                               val_fraction=val_fraction,
                                               return_train_score=return_train_score,
                                               scoring="neg_mean_squared_error", verbose=verbose,
                                               return_estimator=True, n_jobs=n_jobs)
    partitions = partition(Y, n_splits=n_splits, val_fraction=val_fraction)
    test_scores = []
    estimators = train_val_results['estimator']
    scorer = score(scoring)
    for (_, _, test), estimator_ in zip(partitions, estimators):
        _score = scorer(Y[test], estimator_.predict(X[test]))
        test_scores.append(_score)

    return_ = {'test_score': test_scores}
    if return_train_score:
        return_['train_val_info'] = train_val_results
    return return_


def partition(X, n_splits=5, val_fraction=0.1):
    """Partition the data into series of train/validation/test sets"""
    partition_model = TimeSeriesSplit(n_splits=n_splits)
    for train_validation, test in partition_model.split(X):
        train_end = int((1.0 - val_fraction) * len(train_validation))
        train = train_validation[:train_end]
        validation = train_validation[train_end:]
        # train.append()
        # validation.append()
        yield train, validation, test


def _train_validation(partitions):
    for train, val, _ in partitions:
        # print('train: ', train[-1])
        # print('val: ', val)
        # print('test: ', _)
        yield train, val


def bias_variance(insample_error, outsample_error, return_bias=False, return_variance=False):
    bias = insample_error
    variance = np_abs(insample_error - outsample_error)
    _bias_variance = bias + variance
    _return = {'bias_variance': _bias_variance}
    if return_bias:
        _return['bias'] = bias
    if return_variance:
        _return['variance'] = variance
    return _return
