from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy import sqrt as np_sqrt


def mse(y_true, y_pred, multioutput='raw_values'):
	return mean_squared_error(y_true, y_pred, multioutput=multioutput)


def rmse(y_true, y_pred, multioutput='raw_values'):
	return np_sqrt(mse(y_true, y_pred, multioutput=multioutput))
	
	
def mape(y_true, y_pred):
	pass
	

def mae(y_true, y_pred, multioutput='uniform_average'):
	return mean_absolute_error(y_true, y_pred, multioutput=multioutput)
	
	
def mean_squared_log_error(y_true, y_pred):
	pass
	

def mean_median_absolute_error(y_true, y_pred):
	pass
	

def r2(y_true, y_pred):
	return r2_score(y_true, y_pred)
