import numpy as np
from sklearn.metrics import r2_score


def slidingwindowsegment(sequence, create_segment, max_error, seq_range=None):
    """
    Return a list of line segments that approximate the sequence.

    The list is computed using the sliding window technique. 

    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two arguments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error

    """
    if not seq_range:
        seq_range = (0, len(sequence)-1)

    start = seq_range[0]
    end = start
    result_segment, _ = create_segment(sequence, (seq_range[0], seq_range[1]))
    while end < seq_range[1]:
        end += 1
        test_segment, error = create_segment(sequence, (start, end))
        if error <= max_error:
            result_segment = test_segment
        else:
            break

    if end == seq_range[1]:
        return [result_segment]
    else:
        return [result_segment] + slidingwindowsegment(sequence, create_segment, max_error, (end-1,seq_range[1]))


def bottomupsegment(sequence, create_segment, max_error):
    """
    Return a list of line segments that approximate the sequence.

    The list is computed using the bottom-up technique.

    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error

    """
    segments = []
    for x0, x1 in zip(range(len(sequence))[:-1], range(len(sequence))[1:]):
        segment, _ = create_segment(sequence, (x0, x1))
        segments.append(segment)
    print("initial segments created")
    if max_error == 0:
        print("Max error is 0, returning initial segments")
        return segments
    mergesegments, mergecosts = [], []
    for seg1, seg2 in zip(segments[:-1], segments[1:]):
        mergesegment, mergecost = create_segment(sequence, (seg1[0], seg2[2]))
        mergesegments.append(mergesegment)
        mergecosts.append(mergecost)
    print("initial segments merged and merge costs calculated")

    min_cost = min(mergecosts)
    while min_cost < max_error:
        # print("Min Cost: ", min_cost)
        idx = mergecosts.index(min_cost)
        segments[idx] = mergesegments[idx]
        del segments[idx+1]

        if idx > 0:
            x0, x1 = segments[idx-1][0], segments[idx][2]
            mergesegments[idx-1], mergecosts[idx-1] = create_segment(sequence, (x0, x1))

        if idx+1 < len(mergecosts):
            x0, x1 = segments[idx][0], segments[idx+1][2]
            mergesegments[idx+1], mergecosts[idx+1] = create_segment(sequence, (x0, x1))

        del mergesegments[idx]
        del mergecosts[idx]

        min_cost = min(mergecosts)

    return segments


class Segment:
    def __init__(self, l, r, mc):
        self.left = l
        self.right = r
        self.merging_cost = mc

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_merging_cost(self):
        return self.merging_cost

    def set_right(self, l):
        self.left = l

    def set_right(self, r):
        self.right = r

    def set_merging_cost(self, mc):
        self.merging_cost = mc


def best_fit_error(data, start, end):
    indices = np.array(range(start, end))
    (p, residuals, rank, singular_values, rcond) = np.polyfit(indices, data, deg=1, full=True)
    return residuals[0]


def interpolation_error(data, start, end):
    slope = (data[-1] - data[0]) / (end - 1 - start)
    indices = np.arange(start, end)
    estimated_y = slope * indices + data[0]
    return r2_score(data, estimated_y)


def get_segments(data, num_segments, slope_estimator='interpolation'):
    if slope_estimator == 'interpolation':
        slope_estimator_error = interpolation_error
    else:
        slope_estimator_error = best_fit_error
    left_indices = range(0, len(data) - 1)
    right_indices = range(1, len(data))
    max_num_segments = len(left_indices)  # OR len(data)/2
    segments = []
    # Create initial fine approximation
    for i in range(max_num_segments):
        segments.append(Segment(left_indices[i], right_indices[i], float('inf')))

    # Find the cost of merging each pair of segments each the last segment which doesn't have a next segment
    for i in range(max_num_segments - 1):
        start, end = segments[i].get_left(), segments[i + 1].get_right() + 1
        target = np.array(data[start: end])
        cost = slope_estimator_error(target, start, end)
        segments[i].set_merging_cost(cost)

    # Merge adjacent segments with least merging cost
    while len(segments) > num_segments:
        # print('segments: ', len(segments), "/", num_segments)
        # Find index of the segment with the smallest merging cost
        costs = [segment.merging_cost for segment in segments]
        min_index = np.argmin(costs)
        # Update the merging cost of the new segment
        if min_index >= len(segments) - 2:
            # upper boundary, set cost to infinity
            segments[min_index].set_merging_cost(float('inf'))
        else:
            start, end = segments[min_index].get_left(), segments[min_index + 2].get_right() + 1
            target = np.array(data[start: end])
            cost = slope_estimator_error(target, start, end)
            segments[min_index].set_merging_cost(cost)

        # Update new right index
        new_right_index = segments[min_index + 1].get_right()
        segments[min_index].set_right(new_right_index)

        # Delete absorbed segment
        segments.remove(segments[min_index + 1])

        # Update the merging cost of the previous segment to new (merged) segment if necessary
        if min_index != 0:
            start, end = segments[min_index - 1].get_left(), segments[min_index].get_right() + 1
            target = np.array(data[start: end])
            cost = slope_estimator_error(target, start, end)
            segments[min_index - 1].set_merging_cost(cost)

    return segments


def format_segments(segments, data):
    for segment in range(len(segments)):
        x0 = segments[segment].get_left()
        x1 = segments[segment].get_right()
        segments[segment] = (x0, data[x0], x1, data[x1])
    return segments

    
def topdownsegment(sequence, create_segment, max_error, seq_range=None):
    """
    Return a list of line segments that approximate the sequence.
    
    The list is computed using the top-down technique.
    
    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two arguments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error
    
    """
    if not seq_range:
        seq_range = (0, len(sequence)-1)

    bestlefterror, bestleftsegment = float('inf'), None
    bestrighterror, bestrightsegment = float('inf'), None
    bestidx = None

    for idx in range(seq_range[0]+1, seq_range[1]):
        segment_left, error_left = create_segment(sequence, (seq_range[0], idx))
        segment_right, error_right = create_segment(sequence, (idx, seq_range[1]))
        if error_left + error_right < bestlefterror + bestrighterror:
            bestlefterror, bestrighterror = error_left, error_right
            bestleftsegment, bestrightsegment = segment_left, segment_right
            bestidx = idx
    
    if bestlefterror <= max_error:
        leftsegs = [bestleftsegment]
    else:
        leftsegs = topdownsegment(sequence, create_segment, max_error, (seq_range[0],bestidx))
    
    if bestrighterror <= max_error:
        rightsegs = [bestrightsegment]
    else:
        rightsegs = topdownsegment(sequence, create_segment, max_error, (bestidx,seq_range[1]))
    
    return leftsegs + rightsegs
