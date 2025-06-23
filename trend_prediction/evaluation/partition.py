from numpy import arange as np_arange


class TimeSeriesPartition:
    def __init__(self, data, train_size, test_size, shift_train=True, validation=True):
        self.train_size = train_size
        self.test_size = test_size
        self.validation = validation
        self.shift_train = shift_train
        self.n_samples = len(data)
        if validation:
            self.n_splits = (self.n_samples - self.train_size - self.test_size) // self.test_size
        else:
            self.n_splits = (self.n_samples - self.train_size) // self.test_size

    def split(self):
        indices = np_arange(self.n_samples)
        train_start = 0
        for split in range(self.n_splits):
            if self.shift_train:
                train_start = split * self.test_size
            train_end = self.train_size + split * self.test_size
            train = indices[train_start: train_end]
            if self.validation:
                validation_start = train_end
                validation_end = train_end + self.test_size
                validation = indices[validation_start: validation_end]
                test = indices[validation_end: validation_end + self.test_size]
                yield train, validation, test
            else:
                test = indices[train_end: train_end + self.test_size]
                yield train, test


if __name__ == '__main__':
    from numpy import array
    data = array([0,1,2,3,4,5,6,7,8,9])
    spliter = TimeSeriesPartition(data, 2, 1, False, True)
    for train, val, test in spliter.split():
        # print('train: ', train)
        # print('test: ', test)
        print('train: ', data[train])
        print('val: ', data[val])
        print('test: ', data[test])
