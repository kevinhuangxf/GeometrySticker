import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    print("mode dict: ", model_dict.keys())
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    print(f'Loading checkpoint from {checkpoint_.keys()}')
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict, strict=False)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input,(2,2,2,2),mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)

def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def sharpen(data, label_en, LABEL_SHARP=1.0):
    batchsize, c, h, w = data.shape
    # data_flat = data.view(batchsize, -1)
    data_flat = data.reshape(batchsize, -1)
    mean_ = torch.mean(data_flat, -1)
    max_, _ = torch.max(data_flat, -1)
    min_, _ = torch.min(data_flat, -1)

    min_ = min_.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    min_ = min_.repeat(1, c, h, w)

    max_ = max_.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    max_ = max_.repeat(1, c, h, w)

    interval = max_ - min_
    data_norm = (data - min_) / (interval + 1e-6)

    # label_en = GaussianSmoothing(1, 5, 1)
    data_smooth = label_en(data_norm)
    data_sharpen = data_smooth + \
        (1.0 + LABEL_SHARP) * (data_norm - data_smooth)

    data_sharpen = data_sharpen * interval + min_

    data_sharpen_flat = data_sharpen.view(batchsize, -1)
    _mean = torch.mean(data_sharpen_flat, -1)
    _max, _ = torch.max(data_sharpen_flat, -1)
    _min, _ = torch.min(data_sharpen_flat, -1)

    _min = _min.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    _min = _min.repeat(1, c, h, w)

    _max = _max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    _max = _max.repeat(1, c, h, w)

    data_sharpen_norm = (data_sharpen - _min) / (_max - _min + 1e-6)

    data_sharpen_norm = data_sharpen_norm * interval + min_

    return data_sharpen
