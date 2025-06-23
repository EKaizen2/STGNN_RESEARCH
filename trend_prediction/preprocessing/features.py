import pandas as pd
import numpy as np
from preprocessing import segment, fit


def moving_average(data, window_size):
	return pd.DataFrame(data).rolling(window_size).mean().iloc[window_size - 1:]


def _get_lags(data, lag=30, stride=15):
	start_indices = range(0, len(data) - lag + 1, stride)
	end_indices = range(lag, len(data) + 1, stride)
	for start, end in zip(start_indices, end_indices):
		yield (start, end)


def quantize(data, n_bins=90, max_value=90, min_value=-90):
	stride = (max_value - min_value) // n_bins
	boundaries = list(range(min_value, max_value + 1, stride))
	quantised_data = np.zeros(data.shape)
	for label, lower, upper in zip(range(n_bins), boundaries[:-1], boundaries[1:]):
		quantised_data[np.bitwise_and(lower < data, data < upper)] = label
	return quantised_data


def movement_from_last_n(data, lag=30, duration=2, overlap=True,
						 overlap_fraction=0.5, max_feature_size=None, n_trend=True,
						 n_segments=5, angle_metric='degree', quantise=False):
	if overlap:
		stride = int(np.ceil(np.round((1 - overlap_fraction) * lag, 2)))
	else:
		stride = 1
	features = []
	targets = []
	for start, end in _get_lags(data, lag, stride):
		# if no more data for target, continue
		if end < len(data):
			if data[end] < data[end - 1]:
				targets.append(0)
			else:
				targets.append(1)
		else:
			continue
		segment = data[start: end]
		features.append(np.array(segment))
		# if n_trend:
			# features.append(np.array(segment))
			# segments = _get_segments(segment, n_segments, 'regression')
			# trendlines = _unspecified_duration_trend(segments, angle_metric=angle_metric)
			# if max_feature_size is None:
			# 	max_feature_size = trendlines.shape[0]
			# starting_index = trendlines.shape[0] - max_feature_size
			# if quantise:
			# 	features.append(quantize(trendlines['strength', 'duration'].values[starting_index:]))
			# else:
			# 	features.append(trendlines[['strength', 'duration']].values[starting_index:])
			# features[-1] = np.concatenate((np.array(features[-1]).reshape(1, -1),
			# 							   np.array(segment).reshape(1, -1)), axis=1)
		# else:
		# 	movement_starts = list(range(0, len(segment) - duration + 1, duration - 1))
		# 	movement_ends = list(range(duration - 1, len(segment), duration - 1))
		# 	movement_starts.reverse()
		# 	movement_ends.reverse()
		# 	row = []
		# 	if max_feature_size is None:
		# 		max_feature_size = len(movement_starts)
		# 	for movement_start, movement_end in zip(movement_starts[:max_feature_size], movement_ends[:max_feature_size]):
		# 		if segment[movement_end] < segment[movement_start]:
		# 			feature = 0
		# 		else:
		# 			feature = 1
		# 		row.insert(0, feature)
		# 	features.append(row)
	return np.array(features).reshape((len(features), -1)),\
		   np.array(targets).reshape(len(features), )


def movement_direction(strength, n_classes=3, lower=-0.5, upper=0.55):
	directions = np.ones(len(strength)) * -1
	for idx, value in enumerate(strength):
		if n_classes == 3:
			if value <= lower:
				directions[idx] = 0
			elif value < upper:
				directions[idx] = 1
			else:
				directions[idx] = 2
		elif n_classes == 2:
			if value < lower:
				directions[idx] = 0
			else:
				directions[idx] = 1
	return directions


def local_data(raw_data, trend_indices, trend_window=1, local_window=100):
	trend_indices = trend_indices[trend_window:].astype(int)
	window = min(local_window, trend_indices[0] + 1)
	_local_data = np.ndarray((len(trend_indices), window))
	for idx, trend_index in enumerate(trend_indices):
		_local_data[idx] = raw_data[trend_index - window + 1:trend_index + 1]
	return _local_data


def trend(data, strength_metric='angle', angle_metric=None, duration=None,
          return_segment=False, overlap=False, overlap_fraction=0.0,
		  pla_algorithm='bottom', slope_estimator='regression',
		  error=False, max_error=10, n_segments=4418):
	slope_estimator = slope_estimator.lower()
	if duration is None:
		if slope_estimator == 'regression':
			line_fitter = fit.regression
		elif slope_estimator == 'interpolation':
			line_fitter = fit.interpolate
		if pla_algorithm == 'bottom':
			if error:
				segments = _bottom_up(data, max_error=max_error, line_fitter=line_fitter)
			else:
				segments = _get_segments(data, n_segments, slope_estimator=slope_estimator)
		elif pla_algorithm == 'top':
			segments = _top_down(data, max_error=max_error, line_fitter=line_fitter)
		elif pla_algorithm == 'sliding_window':
			segments = _sliding_window(data, max_error=max_error, line_fitter=line_fitter)
		trends = _unspecified_duration_trend(segments, angle_metric)
		segments = pd.DataFrame(segments, columns=['start_x', 'end_x', 'start_x', 'end_y'])
	else:
		trends, segments = _specified_duration_trend(data, duration, strength_metric=strength_metric,
													 angle_metric=angle_metric, overlap=overlap,
													 overlap_fraction=overlap_fraction, line_fitter=slope_estimator)
	if return_segment:
		return trends, segments
	else:
		return trends


def _unspecified_duration_trend(segments, angle_metric):
	trends = np.zeros((len(segments), 3))
	for idx in range(len(segments)):
		x0, y0, x1, y1 = segments[idx]
		if angle_metric == 'radian':
			slope = np.arctan((y1 - y0) / (x1 - x0))
		elif angle_metric == 'degree':
			slope = np.arctan((y1 - y0) / (x1 - x0)) * 180.0 / np.pi
		else:
			slope = 1.0*(y1 - y0)/(x1 - x0)
		duration = x1 - x0 + 1
		intercept_index = x0
		trends[idx] = [slope, duration, intercept_index]
	return pd.DataFrame(trends, columns=['strength', 'duration', 'index'])


def interpolate(data, start_end):
	return ((data[start_end[1]] - data[start_end[0]]) / (start_end[1] - start_end[0]), None), None


def _specified_duration_trend(data, duration, strength_metric='angle', angle_metric='radian',
							  overlap=False, overlap_fraction=2/3, line_fitter='regression'):
	print(data.shape)
	if overlap:
		stride = int(np.ceil(np.round((1 - overlap_fraction) * duration, 2)))
		start_indices = range(0, len(data) - duration + 1, stride)
		end_indices = range(duration - 1, len(data), stride)
	else:
		start_indices = range(0, len(data) - duration + 1, duration - 1)
		end_indices = range(duration - 1, len(data), duration - 1)
	gradients = []
	if strength_metric == 'angle':
		print("angle")
		if line_fitter == "interpolation":
			line_fitter = interpolate
		elif line_fitter == "regression":
			line_fitter = fit.leastsquareslinefit
		for start, end in zip(start_indices, end_indices):
				gradient_intercept, _ = line_fitter(data, (start, end))
				gradients.append(gradient_intercept[0])
		gradients = np.array(gradients)
		if angle_metric == 'radian':
			strengths = np.arctan(gradients)
		elif angle_metric == 'degree':
			strengths = np.arctan(gradients) * 180.0 / np.pi
		else:
			strengths = gradients
	elif strength_metric == 'percent':
		strengths = 100.0 * (data[end_indices] - data[start_indices]) / data[start_indices]
	segments = pd.DataFrame({'start_x': start_indices, 'start_y': data[start_indices],
							 'end_x': end_indices, 'end_y': data[end_indices]})
	return pd.DataFrame({'strength': strengths, 'index': start_indices}, columns=['strength', 'index']), segments


def _sliding_window(data, max_error, line_fitter):
	return segment.slidingwindowsegment(data, line_fitter, max_error)


def _bottom_up(data, max_error, line_fitter):
	return segment.bottomupsegment(data, line_fitter, max_error)


def _get_segments(data, n_segments, slope_estimator='regression'):
	if n_segments == 1:
		return [(0, data[0], len(data) - 1, data[-1])]
	return segment.format_segments(segment.get_segments(data, n_segments, slope_estimator), data)


def _top_down(data, max_error, line_fitter):
	return segment.topdownsegment(data, line_fitter, max_error)

