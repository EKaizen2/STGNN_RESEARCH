import numpy as np
from numpy import arange, array, ones
from numpy.linalg import lstsq


# create_segment functions
def regression(sequence, seq_range):
    """Return (x0,y0,x1,y1, residual) of a line fit to a segment of a sequence using linear regression"""
    p, residuals = leastsquareslinefit(sequence, seq_range)
    y0 = p[0]*seq_range[0] + p[1]
    y1 = p[0]*seq_range[1] + p[1]
    return (seq_range[0], y0, seq_range[1], y1), residuals

    
def interpolate(sequence, seq_range):
    """Return (x0,y0,x1,y1, residual) of a line fit to a segment using a simple interpolation"""
    segment = (seq_range[0], sequence[seq_range[0]], seq_range[1], sequence[seq_range[1]])
    error = rmse(sequence, segment)
    return segment, error


# compute_error functions
def sum_residuals(sequence, segment):
    x0, x1 = segment[0], segment[2]
    y0, y1 = segment[1], segment[3]
    x = np.arange(x0, x1 + 1)
    y = np.array(sequence[x0: x1 + 1])
    gradient = 1.0*(y1 - y0)/(x1 - x0)
    best_fit_sequence = gradient * x + y0
    residuals = np.sum((best_fit_sequence - y) ** 2)
    return residuals


def rmse(sequence, segment):
    x0, x1 = segment[0], segment[2]
    y0, y1 = segment[1], segment[3]
    x = np.arange(x0, x1 + 1)
    y = np.array(sequence[x0: x1 + 1])
    gradient = 1.0*(y1 - y0)/(x1 - x0)
    best_fit_sequence = gradient * x + y0
    rmse = np.sqrt(np.mean((best_fit_sequence - y) ** 2))
    return rmse


def leastsquareslinefit(sequence, seq_range):
    """Return the parameters and error for a least squares line fit of one segment of a sequence"""
    x = arange(seq_range[0], seq_range[1]+1)
    y = array(sequence[seq_range[0]:seq_range[1]+1])
    A = ones((len(x), 2), float)
    A[:, 0] = x
    (p, residuals, rank, s) = lstsq(A, y, rcond=None)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return p, error
