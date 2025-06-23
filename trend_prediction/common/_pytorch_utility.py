import torch


def create_tensor(shape=None):
    if torch.cuda.is_available():
        if shape is None:
            return torch.cuda.FloatTensor()
        return torch.cuda.FloatTensor(shape)
    else:
        if shape is None:
            return torch.FloatTensor()
        return torch.FloatTensor(shape)


def create_tensor_from_numpy(numpy_array):
    if torch.cuda.is_available():
        return torch.from_numpy(numpy_array).type(torch.cuda.FloatTensor)
    else:
        return torch.from_numpy(numpy_array).type(torch.FloatTensor)


def detach_tensor(tensor):
    if tensor.device.type == 'cpu':
        return tensor.detach().numpy()
    return tensor.cpu().detach().numpy()
