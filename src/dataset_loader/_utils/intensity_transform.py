import numpy as np
from skimage.exposure import equalize_adapthist
import torch
from scipy.ndimage import gaussian_filter
import scipy
import random
import torch as th
from PIL import Image
from scipy.interpolate import RectBivariateSpline


class MyRandomImageContrastTransform(object):

    def __init__(self, random_state=None, is_labelmap=[False, True], clip_limit_range=[0.01, 1], nbins=256,
                 enable=False):
        """
        Perform Contrast Limited Adaptive Histogram Equalization (CLAHE)
    .   An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the
    image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
        Based on https://scikit-image.org/docs/dev/api/skimage.exposure.html?highlight=equalize_adapthist#skimage
        .exposure.equalize_adapthist
        Arguments
        ---------

        """
        self.random_state = random_state
        self.clip_limit_range = clip_limit_range  # [0,1] The larger the value, the higher the contrast
        self.nbins = nbins
        self.is_label_map = is_labelmap
        self.enable = enable

    def __call__(self, *inputs):
        if self.enable:
            outputs = []
            assert len(self.is_label_map) == len(
                inputs), 'for each input, must clarify whether this is a label map or not.'
            clip_limit = np.random.uniform(low=self.clip_limit_range[0], high=self.clip_limit_range[1])
            for idx, _input in enumerate(inputs):
                _input = _input.numpy()
                flag = self.is_label_map[idx]
                if flag:
                    result = _input
                else:
                    print(_input.shape)
                    result = np.zeros(_input.shape, dtype=_input.dtype)
                    for i in range(_input.shape[0]):
                        temp = _input[i]
                        print('temp shape', temp.shape)
                        _input_min = temp.min()
                        _input_max = temp.max()
                        ## clahe requires intensity to be Uint16
                        temp = intensity_normalise(temp, perc_threshold=(0., 100.0), min_val=0, max_val=255)
                        temp = np.int16(temp)
                        clahe_output = equalize_adapthist(temp, clip_limit=clip_limit, nbins=self.nbins)
                        ## recover intensity range
                        result[i] = intensity_normalise(clahe_output, perc_threshold=(0., 100.0), min_val=_input_min,
                                                        max_val=_input_max)

                tensorresult = torch.from_numpy(result).float()
                outputs.append(tensorresult)
                return outputs if idx >= 1 else outputs[0]

        else:
            outputs = inputs
            return outputs


class RandomGamma(object):
    '''
    Perform Random Gamma Contrast Adjusting
    support 2D and 3D
    '''

    def __init__(self, p_thresh=0.5, gamma_range=[0.8, 1.4], gamma_flag=True, preserve_range=True):
        """
        Randomly do gamma to a torch tensor

        Arguments
        --------
        :param gamma_flag: [bool] list of flags for gamma aug

        """
        self.gamma_range = gamma_range
        self.p_thresh = p_thresh

        self.gamma_flag = gamma_flag

        self.preserve_range = preserve_range  ##  if preserve the range to be in [min,max]

    def __call__(self, *inputs):
        outputs = []
        if np.random.rand() < self.p_thresh:
            gamma = random.random() * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]  #
            # print ('gamma: %f',gamma)
            for idx, _input in enumerate(inputs):
                assert inputs[0].size() == _input.size()
                if (self.gamma_flag[idx]):
                    assert gamma > 0
                    if self.preserve_range:
                        self.c_min = _input.min()
                        self.c_max = _input.max()
                    _input = _input ** (1.0 / gamma)
                    if self.preserve_range:
                        _input[_input < self.c_min] = self.c_min
                        _input[_input > self.c_max] = self.c_max

                outputs.append(_input)
        else:
            idx = len(inputs)
            outputs = inputs
        return outputs if idx >= 1 else outputs[0]


class RandomBrightnessFluctuation(object):
    '''
    Perform image contrast and brightness augmentation.
    support 2D and 3D
    '''

    def __init__(self, p=0.5, contrast_range=[0.8, 1.2], brightness_range=[-0.1, 0.1], flag=True, preserve_range=True):
        """
        Arguments
        --------
        :param flag: [bool] list of flags for aug

        """
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range

        self.p_thresh = p

        self.flag = flag

        self.preserve_range = preserve_range  ##  if preserve the range to be in [min,max]

    def __call__(self, *inputs):
        outputs = []
        if np.random.rand() < self.p_thresh:
            scale = random.random() * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]  #
            brightness = random.random() * (self.brightness_range[1] - self.brightness_range[0]) + \
                         self.brightness_range[
                             0]  #
            # print ('gamma: %f',gamma)
            for idx, _input in enumerate(inputs):
                assert inputs[0].size() == _input.size()
                if (self.flag[idx]):
                    assert scale > 0
                    if self.preserve_range:
                        self.c_min = _input.min()
                        self.c_max = _input.max()

                    _input = _input * scale + brightness

                    if self.preserve_range:
                        _input[_input < self.c_min] = self.c_min
                        _input[_input > self.c_max] = self.c_max

                outputs.append(_input)
        else:
            idx = len(inputs)
            outputs = inputs
        return outputs if idx >= 1 else outputs[0]


def intensity_normalise(img_data, perc_threshold=(0., 99.0), min_val=0., max_val=1):
    '''
    intensity_normalise
    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    img_data=3D matrix [N*H*W]
    '''
    if len(img_data.shape) == 3:
        output = np.zeros_like(img_data)
        assert img_data.shape[0] < img_data.shape[1], 'check data is formatted as N*H*W'
        for idx in range(img_data.shape[0]):  #
            slice_data = img_data[idx]
            a_min_val, a_max_val = np.percentile(slice_data, perc_threshold)
            ## restrict the intensity range
            slice_data[slice_data <= a_min_val] = a_min_val
            slice_data[slice_data >= a_max_val] = a_max_val
            ## perform normalisation
            scale = (max_val - min_val) / (a_max_val - a_min_val)
            bias = max_val - scale * a_max_val
            output[idx] = slice_data * scale + bias
        return output
    elif len(img_data.shape) == 2:
        a_min_val, a_max_val = np.percentile(img_data, perc_threshold)
        ## restrict the intensity range
        img_data[img_data <= a_min_val] = a_min_val
        img_data[img_data >= a_max_val] = a_max_val
        ## perform normalisation
        scale = (max_val - min_val) / (a_max_val - a_min_val)
        bias = max_val - scale * a_max_val
        output = img_data * scale + bias
        return output

    else:
        raise NotImplementedError


def contrast_enhancement(img_data, clip_limit=0.01, nbins=256):
    if len(img_data.shape) == 3:
        output = np.zeros_like(img_data)
        assert img_data.shape[0] < img_data.shape[1], 'check data is formatted as N*H*W'
        for idx in range(img_data.shape[0]):  #
            slice_data = img_data[idx]
            slice_data = equalize_adapthist(slice_data, clip_limit=clip_limit, nbins=nbins)
            output[idx] = slice_data
        return output
    else:
        raise NotImplementedError


class MyNormalizeMedicPercentile(object):
    """
    Given min_val: float and max_val: float,
    will normalize each channel of the th.*Tensor to
    the provided min and max values.

    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values,
    and min & max are observed min/max for each channel
    """

    def __init__(self,
                 min_val=0.0,
                 max_val=1.0,
                 perc_threshold=(1.0, 95.0),
                 norm_flag=True):
        """
        Normalize a tensor between a min and max value
        :param min_val: (float) lower bound of normalized tensor
        :param max_val: (float) upper bound of normalized tensor
        :param perc_threshold: (float, float) percentile of image intensities used for scaling
        :param norm_flag: [bool] list of flags for normalisation
        """

        self.min_val = min_val
        self.max_val = max_val
        self.perc_threshold = perc_threshold
        self.norm_flag = norm_flag

    def __call__(self, *inputs):
        # prepare the normalisation flag
        if isinstance(self.norm_flag, bool):
            norm_flag = [self.norm_flag] * len(inputs)
        else:
            norm_flag = self.norm_flag

        outputs = []
        eps = 1e-8
        for idx, _input in enumerate(inputs):
            if norm_flag[idx]:
                # determine the percentiles and threshold the outliers
                _min_val, _max_val = np.percentile(_input.numpy(), self.perc_threshold)
                _input[th.le(_input, _min_val)] = _min_val
                _input[th.ge(_input, _max_val)] = _max_val
                # scale the intensity values
                a = (self.max_val - self.min_val) / ((_max_val - _min_val) + eps)
                b = self.max_val - a * _max_val
                _input = _input.mul(a).add(b)

                _min_val, _max_val = np.percentile(_input.numpy(), self.perc_threshold)
            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]


class MyRandomPurtarbation(object):
    """

    """

    def __init__(self,
                 multi_control_points=[2,4,8],
                 max_sigma=16,
                 flag=True,
                 add_noise=True,
                 epsilon=0.01,
                 p=0.5,
                 magnitude=0.3
                 ):
        """
        Running random perturbation on images
        :param multi_control_points: list of number of control points at each scale, by default, only use 4 control
        points.
        :param max_sigma: float, a parameter to control the scale of gaussian filter for smoothness
        :param flag: whether to apply the perturbation to each input in the list
        :param add_noise: boolean: adding random gaussian noise: default: True
        :param epsilon: float, a scalar to control the level of noise, Default: 0.01
        :param p: the probability of performing perturbation. Default: 0.5
        """
        self.multi_control_points = multi_control_points
        self.max_sigma = max_sigma
        self.flag = flag
        self.add_noise = add_noise
        self.epsilon = epsilon
        assert magnitude>=0 and magnitude<1,'magnitude must be in [0,1)'
        self.magnitude=magnitude
        self.p = p

    def __call__(self, *inputs):
        # prepare the perturbation flag
        if isinstance(self.flag, bool):
            flag = [self.flag] * len(inputs)
        else:
            flag = self.flag
        if np.random.rand() >= self.p:
            # do nothing
            return inputs
        else:
            outputs = []
            if isinstance(self.multi_control_points, list):
                self.multi_control_points.sort()
            else:
                raise ValueError
            for idx, input in enumerate(inputs):
                if flag[idx]:
                    _input = input.numpy()
                    if np.abs(np.sum(_input) - 0) > 1e-6:
                        ##random generate bias field
                        ch, h, w = _input.shape[0], _input.shape[1], _input.shape[2]
                        total_bias_field = np.zeros((h, w))

                        ## from coarse grid to fine grid
                        for control_points in self.multi_control_points:
                            assert control_points <= np.min((h,
                                                             w)), 'num of control points at each scale must be ' \
                                                                  'smaller or equal to the original image size'

                            control_points_field = np.float32(np.random.uniform(0, 1, (control_points, control_points)))
                            sigma = control_points * 2.0
                            if sigma > self.max_sigma: sigma = self.max_sigma

                            control_points_field = gaussian_filter(control_points_field, sigma)
                            interp = np.array(
                                Image.fromarray(control_points_field, mode='L').resize((h, w), resample=Image.BICUBIC),
                                dtype=np.float32)
                            interp = interp / (1.0 * interp.sum() * control_points + 1e-12)
                            total_bias_field += interp

                        total_bias_field = gaussian_filter(total_bias_field, self.max_sigma)
                        total_bias_field = (total_bias_field / (
                                1.0 * total_bias_field.sum() + 1e-12)) * h * w  ## should be close to a identity
                        # restrict values to [1-magnitude, 1+magnitude]
                        total_bias_field=np.clip(total_bias_field,1-self.magnitude,1+self.magnitude)
                        ## bias image
                        _input = np.repeat(total_bias_field[np.newaxis, :, :], repeats=ch, axis=0) * _input

                        _min_val = np.min(np.array(_input))
                        _max_val = np.max(np.array(_input))
                        _input = (_input - _min_val) / (_max_val - _min_val + 1e-8)

                        ## add gaussian noise
                        if self.add_noise:
                            noise = np.random.randn(ch, h, w)
                            noise = noise * self.epsilon
                            _input = _input + noise
                            _input = np.clip(_input, 0, 1)
                    else:
                        print('ignore black images')
                    #
                    input = torch.from_numpy(_input).float()
                    # print (input.size())
                outputs.append(input)

            return outputs if idx >= 1 else outputs[0]


class MyRandomPurtarbationV2(object):
    """

    """

    def __init__(self,
                 ms_control_point_spacing=[32],
                 magnitude=0.2,
                 flag=True,
                 add_noise=True,
                 epsilon=0.01,
                 p=0.5,
                 debug=False,
                 spline_dgree=3,
                 spline_smoothness=3,
                 ):
        """
        Running random perturbation on images, perturbation is smoothed using bspline interpolation
        :param ms_control_point_spacing: list of control point spacing at each scale. Prefer to use 5x5
        control points in the coarse grid (images are divided into 4x4).
        :param magnitude: float, control the value range of knots vectors at the initialization stage
        :param flag: whether to apply the perturbation to each input in the list
        :param add_noise: boolean: adding random gaussian noise: default: True
        :param epsilon: float, a scalar to control the level of noise, Default: 0.01
        :param spline_dgree: int,degree of bivariate spline, default =3
        :param p: the probability of performing perturbation. Default: 0.5
        """
        assert len(ms_control_point_spacing) >= 1, 'must specify at least one spacing, but got {}'.format(
            str(ms_control_point_spacing))
        assert np.abs(magnitude)<1, 'must set magnitude x in a reasonable range, bias field value 1+/-magnitude can not be zero or negative'

        self.ms_control_point_spacing = [64]
        self.magnitude = magnitude
        self.flag = flag
        self.add_noise = add_noise
        self.epsilon = epsilon
        self.spline_dgree = spline_dgree
        self.spline_smoothness = spline_smoothness
        self.p = p
        self.debug = False

    def __call__(self, *inputs):
        # prepare the perturbation flag
        if isinstance(self.flag, bool):
            flag = [self.flag] * len(inputs)
        else:
            flag = self.flag
        if np.random.rand() >= self.p:
            # do nothing
            return inputs
        else:
            outputs = []
            if isinstance(self.ms_control_point_spacing, list):
                ## from coarse to fine:
                self.ms_control_point_spacing.sort(reverse=True)
                if not self.ms_control_point_spacing[-1] == 1:
                    self.ms_control_point_spacing.append(1)
                    self.ms_control_point_spacing.sort(reverse=True)
            else:
                raise ValueError
            for idx, input in enumerate(inputs):
                if flag[idx]:
                    _input = input.numpy()
                    if np.abs(np.sum(_input) - 0) > 1e-6:
                        ##random generate bias field
                        ch, orig_h, orig_w = _input.shape[0], _input.shape[1], _input.shape[2]
                        assert orig_h == orig_w, 'currently only support square images for simplicity, but found size ({},' \
                                       '{})'.format(
                            orig_h, orig_w)
                        raw_image = _input.copy()
                        ## extend the coordinates to be larger than the original
                        h=np.round(orig_h+self.ms_control_point_spacing[0]*1.5)
                        w=np.round(orig_w+self.ms_control_point_spacing[0]*1.5)
                        h=np.int(h)
                        w=np.int(w)
                        assert np.round(h /self.ms_control_point_spacing[0]) >= self.spline_dgree + 1 and np.round(w / self.ms_control_point_spacing[
                                   0]) >= self.spline_dgree + 1, 'please decrease the spacing, the number of control ' \
                                                                'points in each dimension ' \
                                                                'should be at least kx+1, current bspline order k={}, ' \
                                                                'but found only :{} and {} along each axis'.format(
                            self.spline_dgree, h / self.ms_control_point_spacing[0], w / self.ms_control_point_spacing[0])

                        ## initialize the coarsest grid:
                        xmax, ymax = w // 2, h // 2
                        if self.debug:
                            print (xmax,ymax)
                            print ('self.ms_control_point_spacing[0]',self.ms_control_point_spacing[0])
                        x = np.arange(-xmax, xmax + 1, self.ms_control_point_spacing[0])
                        y = np.arange(-ymax, ymax + 1, self.ms_control_point_spacing[0])

                        knots_matrix = 1 + \
                                       np.float32(np.random.uniform(-np.abs(self.magnitude), np.abs(self.magnitude), (len(y), len(x))))  ##initialize value between [-1-magnitude, 1+magnitude]
                        if self.debug: print('initialize {} points'.format(knots_matrix.shape))
                        y_init = x
                        x_init = y
                        z_init = knots_matrix
                        ## from coarse grid to fine grid
                        for spacing in self.ms_control_point_spacing[1:]:
                            interp_spline = RectBivariateSpline(y_init, x_init, z_init, s=self.spline_smoothness,
                                                                kx=self.spline_dgree, ky=self.spline_dgree)
                            if spacing > 1:
                                x2 = np.arange(-xmax, xmax + 1, spacing)
                                y2 = np.arange(-xmax, xmax + 1, spacing)
                            else:
                                ## the finest resolution
                                x2 = np.arange(-xmax, xmax, spacing)
                                y2 = np.arange(-xmax, xmax, spacing)
                            z2 = interp_spline(y2, x2)
                            z_init = z2
                            x_init = x2
                            y_init = y2

                        total_bias_field = (z_init / (
                                1.0 * z_init.sum() + 1e-12)) * h * w  ## should be close to a identity

                        offset_h=np.int((h-orig_h)//2)
                        offset_w=np.int((w-orig_w)//2)

                        total_bias_field=total_bias_field[offset_h:h-offset_h,offset_w:w-offset_w]

                        total_bias_field=np.clip(total_bias_field,1-self.magnitude,1+self.magnitude)
                        _input = np.repeat(total_bias_field[np.newaxis, :, :], repeats=ch, axis=0) * _input

                        _min_val = np.min(np.array(_input))
                        _max_val = np.max(np.array(_input))
                        _input = (_input - _min_val) / (_max_val - _min_val + 1e-8)

                        ## add gaussian noise
                        noise = np.zeros((ch, h, w))

                        if self.add_noise:
                            noise = np.random.randn(ch, orig_h, orig_w)
                            noise = noise * self.epsilon
                            _input = _input + noise
                            _input = np.clip(_input, 0, 1)

                        ## bias image
                        # print(_input.shape)
                        if self.debug:
                            import matplotlib.pyplot as plt
                            font_size = 5
                            plt.figure(dpi=200, frameon=False)
                            plt.subplot(141)
                            plt.axis('off')
                            plt.title('original image', size=font_size)
                            plt.imshow(raw_image[0], cmap='gray')
                            ##plt.colorbar()

                            plt.subplot(142)
                            plt.imshow(total_bias_field, cmap='jet')
                            plt.axis('off')
                            plt.title('random bias field', size=font_size)
                            plt.colorbar()

                            # plt.colorbar()
                            plt.subplot(143)
                            plt.imshow(noise[0], cmap='gray')
                            plt.axis('off')
                            plt.title('noise', size=font_size)

                            plt.subplot(144)
                            plt.imshow(_input[0], cmap='gray')
                            plt.axis('off')
                            plt.title('biased image', size=font_size)
                            plt.show()


                    else:
                        print('ignore black images')
                    #
                    input = torch.from_numpy(_input).float()
                outputs.append(input)

            return outputs if idx >= 1 else outputs[0]