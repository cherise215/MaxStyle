# Created by cc215 at 27/05/19
# perform elastic transform on 2D image data
# for data augmentation
# Enter steps here


# Function to distort image
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform

import torch

class MyElasticTransform(object):

    def __init__(self, alpha=None,sigma=None,random_state=None,is_labelmap=[True, False], p_thresh =0.5, order=3):
        """
        Perform elastic transform
         Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

      Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        Arguments
        ---------
        Random elastic deformations
        sigma: standard deviation value for the normal distribution of gaussian filters.
        order = order of interpolation, order
        """
        self.random_state= random_state
        self.sigma = sigma #small values = local deformations, large values = large deformations.
        self.alpha =alpha
        self.is_label_map=is_labelmap
        self.p_thresh =p_thresh ## if prob>p_thresh, them perform elastic transformation
        self.order=order

    def gen_deformation_field(self,shape,alpha, sigma):
        if self.random_state is None:
            random_state = np.random.RandomState(None)
        else:
            random_state =self.random_state
        sigma = sigma
        alpha= alpha

        # Make random fields
        dx = random_state.uniform(-1, 1, shape)
        dy = random_state.uniform(-1, 1, shape)
        dx = gaussian_filter(dx, sigma=sigma, mode='constant', cval=0) * alpha
        dy = gaussian_filter(dy, sigma=sigma, mode='constant', cval=0) * alpha

        ## deformation field
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return indices

    def __call__(self,*inputs):
        if abs(self.p_thresh-0)<1e-3 or (not np.random.rand()<self.p_thresh):
            return inputs
        else:
            #print ('perform elastic transformations')
            outputs=[]
            assert len(self.is_label_map) >=len(inputs), 'for each input, must clarify whether this is a label map or not.'
            shape = inputs[0].size() ## C H W
            #print (shape)
            alpha = self.alpha
            sigma = self.sigma

            if self.alpha is None:
                alpha = shape[1] * np.random.uniform(low=1.5,high=2) ## draw a value in-between
            if self.sigma is None:
                sigma = shape[1] *np.random.uniform(low=0.1,high=0.2)*3/4

            indices = self.gen_deformation_field(shape[1:],alpha,sigma)
            for idx, _input in enumerate(inputs):
                _input =_input.numpy()
                #print (_input.shape)
                if len(_input.shape) ==3:
                    _input = _input[0] ## 2D
                flag = self.is_label_map[idx]
                if flag:
                    ## label deformation
                    result = np.zeros(shape[1:],np.uint8)
                    ## for each label map do transformation
                    unique_labels = np.unique(_input)
                    for i, c in enumerate(unique_labels):
                        res_new = map_coordinates((_input == c).astype(float), indices, order=self.order, mode='nearest',
                                                  cval=0.).reshape(shape[1:])
                        result[res_new >= 0.5] = c
                else:
                    ## image deformation
                    result = map_coordinates(_input.astype(float), indices, order=self.order, mode='reflect',
                                             cval=0.).reshape(shape[1:])

                tensorresult =torch.from_numpy(result[None,:,:]).float()
                outputs.append(tensorresult)

            return outputs if idx >= 1 else outputs[0]



class MyElasticTransformCoarseGrid(object):

    def __init__(self, mu=0,sigma=10,random_state=None,is_labelmap=[False, True], p_thresh =0.5):
        """
        Perform elastic transform using 3x3 coarse grid
        reference: "Semi-Supervised and Task-Driven Data Augmentation" 
        Arguments
        ---------

        """
        self.random_state= random_state
        self.sigma = sigma 
        self.mu =mu
        self.is_label_map=is_labelmap
        self.p_thresh =p_thresh ## if prob>p_thresh, them perform elastic transformation

    def gen_deformation_field(self,shape, mu=0, sigma=10,order=3):
        if self.random_state is None:
            random_state = np.random.RandomState(None)
        else:
            random_state =self.random_state
        # Make random fields
        dx = random_state.normal(mu, sigma, 9)
        dx_mat = np.reshape(dx,(3,3))
        dy = np.random.normal(mu, sigma, 9)
        dy_mat = np.reshape(dy,(3,3))
        dx_img = transform.resize(dx_mat, output_shape=(shape[0],shape[1]), order=order,mode='reflect')
        dy_img = transform.resize(dy_mat, output_shape=(shape[0],shape[1]), order=order,mode='reflect')

          ## deformation field
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy_img, (-1, 1)), np.reshape(x + dx_img, (-1, 1))
        return indices


    def __call__(self,*inputs):
        if  np.random.rand()>self.p_thresh:
            return inputs
        else:
            # print ('perform elastic transformations')
            outputs=[]
            assert len(self.is_label_map) ==len(inputs), 'for each input, must clarify whether this is a label map or not.'
            shape = inputs[0].size() ## C H W


            indices = self.gen_deformation_field(shape[1:],mu=self.mu,sigma=self.sigma)
            for idx, _input in enumerate(inputs):
                _input =_input.numpy()
                #print (_input.shape)
                if len(_input.shape) ==3:
                    _input = _input[0] ## 2D
                flag = self.is_label_map[idx]
                if flag:
                    result = np.zeros(shape[1:],np.uint8)
                    ## for each label map do transformation
                    unique_labels = np.unique(_input)
                    for i, c in enumerate(unique_labels):
                        res_new = map_coordinates((_input == c).astype(float), indices, order=3, mode='nearest',
                                                  cval=0.).reshape(shape[1:])
                        result[res_new >= 0.5] = c
                else:
                    result = map_coordinates(_input.astype(float), indices, order=3, mode='reflect',
                                             cval=0.).reshape(shape[1:])

                tensorresult =torch.from_numpy(result[None,:,:]).float()
                outputs.append(tensorresult)

            return outputs if idx >= 1 else outputs[0]
