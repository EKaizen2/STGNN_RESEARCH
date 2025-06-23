from sklearn.base import BaseEstimator, TransformerMixin


class _Estimator(BaseEstimator):
	def __init__(self):
		super().__init__()

	def fit(self, X, Y):
		pass

	def predict(self, X, Y=None):
		pass
