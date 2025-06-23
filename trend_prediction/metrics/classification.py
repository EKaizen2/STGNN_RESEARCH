from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer,\
	precision_recall_fscore_support, matthews_corrcoef, confusion_matrix as confusion_matrix_
import numpy as np


def accuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)
	
	
def report(y_true, y_pred, labels=[0, 1], target_names=['Down', 'Up']):
	results = precision_recall_fscore_support(y_true, y_pred, average='binary')
	return {'Accuracy': accuracy_score(y_true, y_pred),
			'Precision': results[0], 'Recall': results[1],
			'F1-Score': results[2]}


def confusion_matrix(y_true, y_pred):
	return confusion_matrix_(y_true, y_pred)


def mcc(y_true, y_pred):
	return matthews_corrcoef(y_true, y_pred)


def f1(y_true, y_pred):
	return f1_score(y_true, y_pred)


def combined_report_loss(y_true, y_pred):
	results = report(y_true, y_pred)
	results = np.array([results['Accuracy'], results['F1-Score']])
	losses = 1 - results
	return np.mean(losses) + np.std(losses)
	# return np.sqrt(np.mean((1 - results)**2))


def custom_metric():
	return make_scorer(combined_report_loss, greater_is_better=False)


