# -- coding: utf-8 --
import  numpy as np
import  torch
def data_transform(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    # x = x / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

def data_inv_transform(x):
    """
    :param x:
    :return:
    """
    recover_data = x * 0.5 + 0.5
    # recover_data = x * 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((28, 28))
    recover_data = recover_data.detach().numpy()
    return recover_data

def str2bool(v):
    return v.lower() in 'true'

def rgb2gray(rgb):
    rgb_np = rgb.detach().cpu().numpy()
    r, g, b = rgb_np[:, 0], rgb_np[:, 1], rgb_np[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray_tensor = torch.from_numpy(gray)
    # gray_tensor = gray_tensor.to(device)
    return gray_tensor
def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype='float32') / 255
    # print('shape of x:', np.shape(x))
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def merge_images(sources, targets, k=10):
    _, _, h, w = sources.shape
    row = int(np.sqrt(64))
    merged = np.zeros([3, row * h, row * w * 2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()