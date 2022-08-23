import SimpleITK as sitk
import torch
from scipy import ndimage
import math
import cv2
import numpy as np
from skimage import transform as sktform
import torch as th
from torch.autograd import Variable
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
import random
from torchsample.utils import  th_random_choice
from torchsample.transforms.affine_transforms import Rotate
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
from torch import Tensor

def transform2tensor(cPader,image_stack, use_gpu=False, variable=False,requires_grad=False,std_norm=False,use_gaussian=True):
    '''
    transform npy data to torch tensor
    :param cPader:pad image to be divided by 16
    :param img_slice: N*H*W
    :param label_slice: N*H*W
    :return: tensor: N*1*H*W
    '''

    new_image_stack = cPader(image_stack)
    ## normalize data
    if std_norm:
        new_image_stack = new_image_stack * 1.0  ##N*H*W
        new_input_mean = np.mean(new_image_stack, axis=(1, 2), keepdims=True)
        new_image_stack -= new_input_mean
        new_std = np.std(new_image_stack, axis=(1, 2), keepdims=True)
        new_image_stack /= (new_std)

        if use_gaussian:
            new_image_stack_copy=new_image_stack.copy()
            for i in range(new_image_stack.shape[0]):
                new_image_stack_copy[i]=gaussian(new_image_stack[i],1)
            new_image_stack=new_image_stack_copy


    new_image_stack = new_image_stack[:, np.newaxis, :, :]

    ##transform to tensor
    new_image_tensor = torch.from_numpy(new_image_stack).float()

    if variable:
        if use_gpu:
            new_image_tensor = Variable(new_image_tensor.cuda(),requires_grad=requires_grad)
        else:
            new_image_tensor = Variable(new_image_tensor,requires_grad=requires_grad)

    return new_image_tensor



def resample_by_spacing(im, new_spacing, interpolator=sitk.sitkLinear, keep_z_spacing=False):
    '''
    resample by image spacing
    :param im: sitk image
    :param new_spacing: new image spa
    :param interpolator: sitk.sitkLinear, sitk.NearestNeighbor
    :return:
    '''
    new_spacing=np.array(new_spacing)
    old_spacing = np.array(im.GetSpacing())
    for i, item in enumerate(new_spacing):
        if item - (-1.) <= 1e-5:
            new_spacing[i] = im.GetSpacing()[i]

    scaling = np.array(new_spacing) / (1.0 * old_spacing)
    new_size = np.round((np.array(im.GetSize()) / scaling)).astype("int").tolist()
    origin_z = im.GetSize()[2]

    if keep_z_spacing:
        new_size[2] = origin_z
        new_spacing[2] =old_spacing[2]
    # if not keep_z_spacing and new_size[2]==origin_z:
    #     print ('shape along z axis does not change')

    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(im, new_size, transform, interpolator, im.GetOrigin(), new_spacing, im.GetDirection())

def resample_by_shape(im, new_size,new_spacing, interpolator=sitk.sitkLinear):
    '''
    resample by image shape
    :param im: sitk image
    :param new_spacing: new image spa
    :param interpolator: sitk.sitkLinear, sitk.NearestNeighbor
    :return:
    '''
    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(im, new_size, transform, interpolator, im.GetOrigin(), new_spacing, im.GetDirection())


def resample_by_ref(im, refim, interpolator=sitk.sitkLinear):
    '''

    :param im: sitk image object
    :param refim:  sitk reference image object
    :param interpolator: sitk.sitkLinear or  sitk.sitkNearestNeighbor
    :return:
    '''
    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(im, refim, transform, interpolator)


def motion_estimation(label, shift=1):
    '''
    perform inter-plane motion for label data
    input N*H*W or a list of labels
    output N*H*W
    '''
    # if isinstance(label,list):
    #     shape=label[0].shape
    #     assert len(label[0].shape) == 3, 'must be 3D image'
    # else:
    #     shape=label.shape
    #     assert len(label.shape) == 3, 'must be 3D image'
    label2 = np.zeros(label.shape, dtype=np.int32)
    for i in range(label2.shape[0]):
        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [np.clip(np.random.normal(), -3, 3) * shift,
                     np.clip(np.random.normal(), -3, 3) * shift]

        # Apply the affine transformation (rotation (0) + scale (1) + random shift) to each slice
        row, col = label.shape[1:3]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), 0, 1.0)
        M[:, 2] += shift_val

        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :], M[:, :2], M[:, 2], order=0)
    return label2



import datetime
#implement list of images to extract fixed z length of volume
class MyRandomExtract(object):
    def __init__(self, size, extract_type=0):
        """
        Perform  a special crop - one of the four corners or center crop
        Arguments
        ---------
        size : integer
            dimensions of the z axis

        crop_type : integer in {0,1,2,3,4}
            0 = center extract
            1 = random extract
            2 = bottom extract
            3 = top extract
        input: D*H*W
        output: size*H*W

        """
        if extract_type not in {0, 1, 2, 3}:
            raise ValueError('extract_type must be in {0, 1, 2, 3, 4}')
        self.size = size
        assert  isinstance(self.size, int), 'size must be int'
        self.extract_type = extract_type


    def crop(self, input):
        x=input
        if self.extract_type == 0:
            # center crop
            x_diff = (x.size(0) - self.size) / 2.
            x_diff=np.int(x_diff)
            return x [x_diff:x_diff+self.size]
        elif self.extract_type == 1:
            x_diff = abs(x.size(0) - self.size-1)
            #print (x_diff)
            random_shift = np.random.randint(0,x_diff)
            print (random_shift + self.size)
            return x [random_shift:random_shift+self.size]
        elif self.extract_type == 2:
            return x [x.size(0)-self.size:]
        elif self.extract_type == 3:
            return x[:self.size]
        else:
            raise NotImplementedError


    def __call__(self,*inputs):
        outputs=[]
        seed= datetime.datetime.now().second + datetime.datetime.now().microsecond
        self.seed=seed
        np.random.seed(seed)

        for idx, _input in enumerate(inputs):
            x = _input
            x = self.crop(x)
            outputs.append(x)
        return outputs if idx >= 1 else outputs[0]


##implement list of images to flip
class MyRandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p

        v : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, *inputs):
        input_dims = len(inputs[0].size())
        h_random_p=random.random()
        v_random_p=random.random()
        outputs = []
        for idx, _input in enumerate(inputs):
            _input= _input.numpy() ##C*H*W
        # horizontal flip with p = self.p
            if self.horizontal:
                if h_random_p< self.p:
                    _input = _input.swapaxes(2, 0) ## W*H*C
                    _input= _input[::-1, ...]
                    _input = _input.swapaxes(0, 2)

            # vertical flip with p = self.p
            if self.vertical:
                if v_random_p < self.p:
                    _input = _input.swapaxes(1, 0)
                    _input = _input[::-1, ...]
                    _input = _input.swapaxes(0, 1)
            input_tensor=torch.from_numpy(_input.copy()) ##convert back to tensor
            outputs.append(input_tensor)
        return outputs if idx >= 1 else outputs[0]

class MySpecialCrop(object):

    def __init__(self, size, crop_type=0):
        """
        Perform a special crop - one of the four corners or center crop

        Arguments
        ---------
        size : tuple or list
            dimensions of the crop

        crop_type : integer in {0,1,2,3,4}
            0 = center crop
            1 = top left crop
            2 = top right crop
            3 = bottom right crop
            4 = bottom left crop
        """
        if crop_type not in {0, 1, 2, 3, 4}:
            raise ValueError('crop_type must be in {0, 1, 2, 3, 4}')
        self.size = size
        self.crop_type = crop_type

    def __call__(self,*inputs):
        indices=None
        input_dims =None
        outputs=[]
        for idx, _input in enumerate(inputs):
            x = _input
            if idx==0:
                ##calc crop position
                input_dims=len(x.size())
                if self.crop_type == 0:
                    # center crop
                    x_diff = (x.size(1) - self.size[0]) / 2.
                    y_diff = (x.size(2) - self.size[1]) / 2.
                    ct_x = [int(math.ceil(x_diff)), x.size(1) - int(math.floor(x_diff))]
                    ct_y = [int(math.ceil(y_diff)), x.size(2) - int(math.floor(y_diff))]
                    indices = [ct_x, ct_y]
                    if input_dims == 4:
                        z_diff = (x.size(3) - self.size[2]) / 2.
                        ct_z = [int(math.ceil(z_diff)), x.size(3) - int(math.floor(z_diff))]
                        indices.append(ct_z)
                elif self.crop_type == 1:
                    # top left crop
                    tl_x = [0, self.size[0]]
                    tl_y = [0, self.size[1]]
                    indices = [tl_x, tl_y]
                    if input_dims == 4:
                        raise NotImplemented
                elif self.crop_type == 2:
                    # top right crop
                    tr_x = [0, self.size[0]]
                    tr_y = [x.size(2) - self.size[1], x.size(2)]
                    indices = [tr_x, tr_y]
                    if input_dims == 4:
                        raise NotImplemented
                elif self.crop_type == 3:
                    # bottom right crop
                    br_x = [x.size(1) - self.size[0], x.size(1)]
                    br_y = [x.size(2) - self.size[1], x.size(2)]
                    indices = [br_x, br_y]
                    if input_dims == 4:
                        raise NotImplemented
                elif self.crop_type == 4:
                    # bottom left crop
                    bl_x = [x.size(1) - self.size[0], x.size(1)]
                    bl_y = [0, self.size[1]]
                    indices = [bl_x, bl_y]
                    if input_dims == 4:
                        raise NotImplemented

            if input_dims == 4:
                x = x[:, indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1]]
            else:
                x = x[:, indices[0][0]:indices[0][1], indices[1][0]:indices[1][1]]
            outputs.append(x)
        return outputs if idx >= 1 else outputs[0]




class MySpecialRandomRotate(object):

    def __init__(self,
                 rotation_range,
                 interp='bilinear',
                 lazy=False,crop=False,desired_size=(1,256,256)):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Before oupput, clip all black borders

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.rotation_range = rotation_range
        self.interp = interp
        self.lazy = lazy
        self.crop=crop
        self.output_size=desired_size

    def __call__(self, *inputs):
        degree = random.uniform(-self.rotation_range, self.rotation_range)

        if self.lazy:
            return MyRotate(degree,interp=self.interp,lazy=True,crop=self.crop)(inputs[0])
        else:
            outputs = MyRotate(degree,
                             interp=self.interp,crop=self.crop)(*inputs)
            return outputs

from torchsample.utils  import th_affine2d

class MyRotate(object):

    def __init__(self,
                 value,
                 interp='bilinear',fill=0,
                 expand=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.value = value
        self.interp = interp
        self.fill = fill
        self.expand=expand 
    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp       
        outputs = []
        fill   = self.fill
        for idx, _input in enumerate(inputs):
            channels = _input.size(0)
            if isinstance(_input, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            image_height=_input.size(1)
            image_width=_input.size(2)
            resample =InterpolationMode.NEAREST

            if interp[idx] == 'bilinear':
                interpolation =InterpolationMode.BILINEAR
            elif interp[idx] == 'nearest':
                interpolation =InterpolationMode.NEAREST
            output = F.rotate(_input, angle=float(self.value),interpolation=interpolation ,expand=self.expand,center=None,fill=fill)
            outputs.append(output)
        return outputs if idx >= 1 else outputs[0]

class MyResize(object):
    """
    resize  a 2D numpy array using skimage , support float type
    ref:http://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
    """

    def __init__(self, size, interp=None, mode='symmetric'):
        self.size = size
        self.mode = mode
        self.order_list=[]
        if isinstance(interp,list):
            for it in interp:
                if it=='bilinear':
                    self.order_list.append(3)
                else:
                    self.order_list.append(0)
        else:
            if interp == 'bilinear':
                self.order_list.append(3)
            else:
                self.order_list.append(0)



    def __call__(self, *input):
        outputs = []
        for idx, _input in enumerate(input):
            x = _input
            x = x.numpy()
            x=x[0,:,:]
            x = sktform.resize(x, output_shape=self.size, order=self.order_list[idx],
                                              mode=self.mode, cval=0, clip=True, preserve_range=True)

            tensor = th.from_numpy(x[np.newaxis,:,:])
            outputs.append(tensor)
        return outputs if idx >= 1 else outputs[0]

class MyPad(object):

    def __init__(self, size):
        """
        Pads an image to the given size

        Arguments
        ---------
        size : tuple or list
            size of crop
        """
        self.size = size

    def __call__(self, *inputs):
        outputs=[]
        for idx, _input in enumerate(inputs):
            x=_input
            x = x.numpy()
            if idx==0:
                shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(x.shape,self.size)]
                shape_diffs = np.maximum(shape_diffs,0)
                pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
            x = np.pad(x, pad_sizes, mode='constant')
            tensor=th.from_numpy(x)
            outputs.append(tensor)
        return outputs if idx >= 1 else outputs[0]



def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


class CropPad(object):
    def __init__(self,h,w,chw=False):
        '''
        if image > taget image size, simply cropped
        otherwise, pad image to target size.
        :param h: target image height
        :param w: target image width
        '''
        self.target_h = h
        self.target_w = w
        self.chw=chw

    def __call__(self,img):
        # center padding/cropping
        if len(img.shape)==3:
            if self.chw:
                x,y=img.shape[1],img.shape[2]
            else:
                x,y=img.shape[0],img.shape[1]
        else:
            x, y = img.shape[0], img.shape[1]

        x_s = (x - self.target_h) // 2
        y_s = (y - self.target_w) // 2
        x_c = (self.target_h - x) // 2
        y_c = (self.target_w - y) // 2
        if len(img.shape)==2:

            if x>self.target_h and y>self.target_w :
                slice_cropped = img[x_s:x_s + self.target_h , y_s:y_s + self.target_w]
            else:
                slice_cropped = np.zeros((self.target_h, self.target_w), dtype=img.dtype)
                if x<=self.target_h and y>self.target_w:
                    slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + self.target_w]
                elif x>self.target_h>0 and y<=self.target_w:
                    slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + self.target_h, :]
                else:
                    slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(img.shape)==3:
            if not self.chw:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[x_s:x_s + self.target_h, y_s:y_s + self.target_w, :]
                else:
                    slice_cropped = np.zeros((self.target_h, self.target_w, img.shape[2]), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[x_c:x_c + x, :, :] = img[:, y_s:y_s + self.target_w, :]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:, y_c:y_c + y, :] = img[x_s:x_s + self.target_h, :, :]
                    else:
                        slice_cropped[x_c:x_c + x, y_c:y_c + y, :] = img
            else:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[:,x_s:x_s + self.target_h, y_s:y_s + self.target_w]
                else:
                    slice_cropped = np.zeros((img.shape[0],self.target_h, self.target_w), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[:,x_c:x_c + x, :] = img[:,:, y_s:y_s + self.target_w]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:,:, y_c:y_c + y] = img[:,x_s:x_s + self.target_h, :]
                    else:
                        slice_cropped[:,x_c:x_c + x, y_c:y_c + y] = img


        return slice_cropped


    def __repr__(self):
        return self.__class__.__name__ + 'padding to ({0}, {1})'. \
            format(self.target_h, self.target_w)




class ReverseCropPad(object):
    def __init__(self,h,w):
        '''
        :param h: original image height
        :param w: original image width
        '''
        self.h = h
        self.w = w

    def __call__(self,slices_cropped):
        if len(slices_cropped.shape)==2:
            # input H*W
            # center padding/cropping
            target_h, target_w = slices_cropped.shape[0], slices_cropped.shape[1]
            result_stack = np.zeros(( self.h, self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[ x_s:x_s + target_h, y_s:y_s + target_w] = slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:, y_s:y_s + target_w] = slices_cropped[x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[x_s:x_s + target_h, :] = slices_cropped[ :, y_c:y_c + self.w]
                else:
                    result_stack = slices_cropped[ x_c:x_c + self.h, y_c:y_c + self.w]

        elif len(slices_cropped.shape)==3:
            # input N*H*W
            # center padding/cropping
            target_h,target_w = slices_cropped.shape[1],slices_cropped.shape[2]
            result_stack=np.zeros((slices_cropped.shape[0],self.h,self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[:,x_s:x_s + target_h , y_s:y_s + target_w]=slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:,:, y_s:y_s + target_w]=slices_cropped[:,x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[:,x_s:x_s + target_h, :]=slices_cropped[:, :,y_c:y_c + self.w]
                else:
                    result_stack=slices_cropped[:,x_c:x_c + self.h, y_c:y_c + self.w]
        elif len(slices_cropped.shape) == 4:
            # input N*C*H*W
            # center padding/cropping
            target_h, target_w = slices_cropped.shape[2], slices_cropped.shape[3]
            result_stack = np.zeros((slices_cropped.shape[0], slices_cropped.shape[1],self.h, self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[:, :,x_s:x_s + target_h, y_s:y_s + target_w] = slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:, :,:, y_s:y_s + target_w] = slices_cropped[:,:, x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[:,:, x_s:x_s + target_h, :] = slices_cropped[:,:, :, y_c:y_c + self.w]
                else:
                    result_stack = slices_cropped[:, :,x_c:x_c + self.h, y_c:y_c + self.w]

        return result_stack

    def __repr__(self):
        return self.__class__.__name__ + 'recover to ({0}, {1})'. \
            format(self.h, self.w)



class NormalizeMedic(object):
    """
    Normalises given slice/volume to zero mean
    and unit standard deviation.
    """

    def __init__(self,
                 norm_flag=True):
        """
        :param norm_flag: [bool] list of flags for normalisation
        """
        self.norm_flag = norm_flag

    def __call__(self, *inputs):
        # prepare the normalisation flag
        if isinstance(self.norm_flag, bool):
            norm_flag = [self.norm_flag] * len(inputs)
        else:
            norm_flag = self.norm_flag

        outputs = []
        for idx, _input in enumerate(inputs):
            if norm_flag[idx]:
                # subtract the mean intensity value
                mean_val = np.mean(_input.numpy().flatten())
                _input = _input.add(-1.0 * mean_val)

                # scale the intensity values to be unit norm
                std_val = np.std(_input.numpy().flatten())
                _input = _input.div(float(std_val))

            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]





class MyRandomChoiceRotate(object):

    def __init__(self,
                 values,
                 p=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image from a list of values. If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        values : a list or tuple
            the values from which the rotation value will be sampled

        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
     
        self.values = values
        if not p is  None:
            p = np.array(self.p)
            if np.abs(1.0-np.sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        else:
            if len(values)>=1:
                p = np.ones(len(values),dtype=np.float32)/(1.0*len(values))
            else:
                p=np.array(1)
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if len(self.values)==0: return inputs
        degree = np.random.choice(self.values, 1, p=self.p)[0]
        degree = np.float32(degree)
        if len(inputs)==1:
            return MyRotate(degree,interp=self.interp)(inputs[0])
        else:
            outputs = MyRotate(degree,
                             interp=self.interp)(*inputs)
            return outputs




class MyRandomAffine(object):

    def __init__(self,
                 rotation_range=30, translation_range=[0.1,0.1],
                                                      shear_range=None,
                                                      zoom_range=[1,1], interp=('bilinear', 'nearest')):
        """_summary_

        Args:
            rotation_range (int, optional): _description_. Defaults to 30.
            translation_range (list, optional): _description_. Defaults to [0.1,0.1].
            shear_range (list, optional): _description_. Defaults to [0,0].
            zoom_range (_type_, optional): _description_. Defaults to [1,1].
            interp (tuple, optional): _description_. Defaults to ('bilinear', 'nearest').
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.interp = interp

    def __call__(self, *inputs):
        if len(self.interp) == 1:
            interp = [self.interp]*len(inputs)
        else: interp =self.interp
        outputs=[]
        affine_augmentor= T.RandomAffine(degrees=(-self.rotation_range, self.rotation_range), translate=(self.translation_range[0], self.translation_range[1]), 
                shear=self.shear_range, scale = self.zoom_range,fill=0, interpolation=InterpolationMode.NEAREST)                
        ret = None
        for idx, _input in enumerate(inputs):
            channels, height, width = _input.size()
            img_size = [width, height]  # flip for keeping BC on get_params call
            fill = affine_augmentor.fill
            if isinstance(_input, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            if ret is None:  ret = affine_augmentor.get_params(affine_augmentor.degrees, affine_augmentor.translate, affine_augmentor.scale, affine_augmentor.shear, img_size)
            if interp[idx] == 'bilinear':
                output = F.affine(_input, *ret, interpolation=InterpolationMode.BILINEAR, fill=fill, center=affine_augmentor.center)
            elif interp[idx] == 'nearest':
                 output = F.affine(_input, *ret, interpolation=InterpolationMode.NEAREST, fill=fill, center=affine_augmentor.center)
            outputs.append(output)
        return outputs if idx >= 1 else outputs[0]