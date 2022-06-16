import numpy as np
import scipy.ndimage
import scipy.misc
import scipy.stats
from skimage import exposure
from skimage.exposure import match_histograms
from .utils import read_saliency


def _gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def process(img, parameter_map):
    """Pre-processes/Post-process saliency images based on options generated from config ParameterMap.

	Args:
        img: a numpy array containing the model output.
        parameter_map: A dict generated by config's ParameterMap.

    Returns:
        result: numpy array
    """
    
    histogram_matching = parameter_map.get_val('histogram_matching')
    histogram_matching_bias_path = parameter_map.get_val('histogram_matching_bias_path')

    center_prior = parameter_map.get_val('center_prior')
    center_prior_prop = parameter_map.get_val('center_prior_prop')
    center_prior_scale_first = parameter_map.get_val('center_prior_scale_first')
    center_prior_weight = parameter_map.get_val('center_prior_weight')

    do_smoothing = parameter_map.get_val('do_smoothing')
    smooth_size = parameter_map.get_val('smooth_size')
    smooth_std = parameter_map.get_val('smooth_std')
    smooth_prop = parameter_map.get_val('smooth_prop')

    scale_output = parameter_map.get_val('scale_output')
    scale_min = parameter_map.get_val('scale_min')
    scale_max = parameter_map.get_val('scale_max')

    if histogram_matching in ('equalization', 'image-based', 'biased'):
        #print("Doing histogram matching")
        if histogram_matching == 'equalization':
            img = exposure.equalize_hist(img)
        elif histogram_matching == 'biased':
            bias = read_saliency(histogram_matching_bias_path)
            img = match_histograms(img, bias)

    if do_smoothing in ('custom', 'proportional'):
        #print("Doing smoothing")
        if do_smoothing == 'custom':
            gauss_filter = _gauss2d(shape=(smooth_size, smooth_size), sigma=smooth_std)
        elif do_smoothing == 'proportional':
            sigma = smooth_prop * max(img.shape)
            gauss_filter = _gauss2d(shape=(3 * sigma, 3 * sigma), sigma=sigma)

        img = scipy.ndimage.correlate(img, gauss_filter, mode='constant')

    if center_prior in ('proportional_add', 'proportional_mult'):
        #print("Doing center biasing")
        if center_prior_scale_first:
            min_val = img.min()
            max_val = img.max()
            img = ((1.0 / (max_val - min_val)) * (img - min_val))

        w = img.shape[0]
        h = img.shape[1]

        x = np.linspace(-(w // 2), (w - 1) // 2, w)
        y = np.linspace(-(h // 2), (h - 1) // 2, h)

        prior_mask_x = scipy.stats.norm.pdf(x, 0, w * center_prior_prop)
        prior_mask_y = scipy.stats.norm.pdf(y, 0, h * center_prior_prop)
        prior_mask = np.outer(prior_mask_x, prior_mask_y)

        if center_prior == 'proportional_add':
            img = (1.0 - center_prior_weight
                   ) * img + center_prior_weight * prior_mask
        elif center_prior == 'proportional_mult':
            img = (1.0 - center_prior_weight) * img + center_prior_weight * (
                img * prior_mask)

    if scale_output == 'min-max':
        img = np.interp(img, (img.min(), img.max()), (scale_min, scale_max))
    elif scale_output == 'normalized':
        img = (img - img.mean()) / img.std()
    elif scale_output == 'log-density':
        min_val = img.min()
        max_val = img.max()
        img = ((1.0 / (max_val - min_val)) * (img - min_val))
        img = np.log(img / img.sum())

    return img


def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res
