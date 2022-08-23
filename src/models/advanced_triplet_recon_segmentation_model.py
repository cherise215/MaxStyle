# this segmentation model is composed of 2 subnetworks at least, an encoder and an decoder

import itertools

import random
import os
from os.path import join
from tkinter import E
from numpy import True_
from numpy.core.fromnumeric import shape
import torch.nn as nn
import torch
import torch.optim as optim
import gc
import torch.nn.functional as F
import copy
import sys
sys.path.append('../')

from src.models.init_weight import reset_bn, init_weights
from src.models.ebm.encoder_decoder import MyEncoder, MyDecoder, Dual_Branch_Encoder
from src.models.segmentation_models.unet import UnetEncoder, UnetDecoder
from src.models.segmentation_models.unetr import UNETR_Encoder, UNETR_Decoder

from src.models.model_util import get_scheduler, _disable_tracking_bn_stats, replace_bn_with_in, get_normalization_params, get_conv_params, recover_model_w_bn, makeVariable, mask_latent_code_channel_wise, mask_latent_code_spatial_wise
from src.models.custom_loss import CustomNormalizedCrossCorrelationLoss, NGF_Loss, contour_loss, basic_loss_fn, entropy_loss, calc_js_divergece, cross_entropy_2D, TVLoss, normalized_cross_correlation
from src.models.custom_layers import BatchInstanceNorm2d

from src.common_utils.metrics import runningScore
from src.common_utils.basic_operations import construct_input, set_grad, rescale_intensity
from src.common_utils.save import save_testing_images_results
from src.common_utils.data_structure import MaxStack, Dictate
from src.common_utils.uncertainty import cal_batch_entropy_maps

from src.advanced.mixstyle import MixStyle
from src.advanced.mixup import ManifoldMixup
from src.advanced.maxstyle import MaxStyle


class AdvancedTripletReconSegmentationModel(nn.Module):
    def __init__(self, network_type='FCN_16_standard', image_ch=1,
                 learning_rate=1e-4,
                 encoder_dropout=None,
                 decoder_dropout=None,
                 num_classes=4, n_iter=1,
                 checkpoint_dir=None, use_gpu=True, debug=False,
                 rec_loss_type='l2',
                 separate_training=False,
                 class_weights=None,
                 optimizer_type = 'Adam',
                image_size = 192,
                intensity_norm_type = "min_max"

                 ):
        """[summary]

        Args:
            network_type (str): network arch name. Default: FCN_16_standard
            image_ch (int, optional):image channel number. Defaults to 1.
            learning_rate (float, optional): learning rate for network parameter optimization. Defaults to 1e-4.
            encoder_dropout (float, optional): [description]. Defaults to None.
            decoder_dropout (float, optional): [description]. Defaults to None.
            num_classes (int, optional): [description]. Defaults to 4.
            n_iter (int, optional): If set to 1, will use FTN's output as final prediction at test time. If set to 2, will use STN's refinement as the final prediction. Defaults to 1.
            checkpoint_dir (str, optional): path to the checkpoint directory. Defaults to None.
            use_gpu (bool, optional): [description]. Defaults to True.
            debug (bool, optional): [description]. Defaults to False.
        """
        super(AdvancedTripletReconSegmentationModel, self).__init__()
        self.network_type = network_type
        self.image_ch = image_ch
        self.image_size  = image_size
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = num_classes

        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.use_gpu = use_gpu
        self.debug = debug
        self.intensity_norm_type = intensity_norm_type

        # initialization
        self.model = self.get_network(checkpoint_dir=checkpoint_dir)
        self.temperature = 2
        self.optimizers = None
        self.optimizer_type = optimizer_type
        self.reset_all_optimizers()
        self.schedulers_dict=None
        self.set_schedulers(self.optimizers)
        self.running_metric = self.set_running_metric()  # cal iou score during training
        self.rec_loss_type = rec_loss_type
        self.class_weights = class_weights
        self.separate_training = separate_training
        self.latent_code = {}
        self.cur_eval_images = None
        self.cur_eval_predicts = None
        self.cur_eval_gts = None  # N*H*W
        self.cur_time_predicts = {}
        self.loss = 0.

    def build_shape_encoder_decoder(self,network_type):
        if  not 'no_STN' in network_type:
            shape_inc_ch = self.num_classes
            if '16' in network_type: reduce_factor = 4
            elif '64' in network_type: reduce_factor = 1
            else:
                NotImplementedError
            if 'w_recon_image' in network_type or 'w_image' in network_type:
                # add reconstructed image information for shape refinement
                shape_inc_ch = self.num_classes + self.image_ch
            elif 'w_dual_image'in network_type:
                    # add both original image and reconstructed image information for shape refinement
                shape_inc_ch = shape_inc_ch + self.image_ch * 2

            shape_encoder = MyEncoder(input_channel=shape_inc_ch, output_channel=512 // reduce_factor, feature_reduce=reduce_factor,
                                    if_SN=False, encoder_dropout=self.encoder_dropout, norm=nn.BatchNorm2d, act=nn.ReLU())
            shape_decoder = MyDecoder(input_channel=512 // reduce_factor, up_type='NN', output_channel=self.num_classes,
                                    feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d)
        else: shape_encoder,shape_decoder =None, None
        return shape_encoder, shape_decoder
    
    def get_network(self, checkpoint_dir=None):
        '''
        get a network model, if checkpoint dir is not none, load weights from the disk
        return a model dict, with keys:
        'image_encoder':
        'image_decoder':
        'segmentation_decoder':
        ## optional
        'shape_encoder':
        'shape_decoder':
        
        '''
        ##
        network_type = self.network_type
        print('construct {}'.format(network_type))
        shape_inc_ch = self.num_classes
        model_dict = {}
        if self.intensity_norm_type=='min_max':
            image_decoder_last_act = nn.Sigmoid() 
        elif self.intensity_norm_type=='z_score':
            image_decoder_last_act =  F.instance_norm
        else:
            raise NotImplementedError
        if 'z_score' in network_type:
            image_decoder_last_act = F.instance_norm
        elif 'identity' in network_type:
            image_decoder_last_act =None
        if network_type in [
            'FCN_64_standard_no_STN',
            'FCN_16_no_STN','FCN_16_standard_no_STN','FCN_16_standard_no_STN_no_im_recon',
            'FCN_16_no_im_recon',
            'FCN_16_standard_w_dual_image', 'FCN_16_standard_w_recon_image',
            'FCN_16_standard_w_image',
            'FCN_16_standard',
            'FCN_16_standard_w_o_filter',
            'FCN_16_standard_share_code',
            'DS_FCN_16_standard',
            'FCN_16_standard_NN_decoder',
            'FCN_64_standard_no_STN_identity',
            'FCN_64_standard_no_STN_z_score']:
            if '16' in network_type:
                reduce_factor = 4
            elif '64' in network_type:
                reduce_factor=1
            else:
                raise ValueError

            # FTN
            if 'DS_FCN' in  network_type:
                image_encoder = Dual_Branch_Encoder(input_channel=self.image_ch, z_level_1_channel=512 // reduce_factor, z_level_2_channel=512 //
                                                    reduce_factor, feature_reduce=reduce_factor, if_SN=False, encoder_dropout=self.encoder_dropout,
                                                    norm=nn.BatchNorm2d, num_domains=2)
            else:
                image_encoder = Dual_Branch_Encoder(input_channel=self.image_ch, z_level_1_channel=512 // reduce_factor, z_level_2_channel=512 //
                                                    reduce_factor, feature_reduce=reduce_factor, if_SN=False, encoder_dropout=self.encoder_dropout,
                                                    norm=nn.BatchNorm2d, num_domains=1)

          
            segmentation_decoder = MyDecoder(input_channel=512 // reduce_factor, up_type='NN', output_channel=self.num_classes,
                                             feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d)

            model_dict = {'image_encoder': image_encoder,
                    'segmentation_decoder': segmentation_decoder,
                    }
            if not 'no_im_recon' in network_type:
                if 'NN_decoder' in network_type:
                    model_dict['image_decoder'] = MyDecoder(input_channel=512 // reduce_factor, up_type='NN', output_channel=self.image_ch,
                                                feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d, last_act=image_decoder_last_act)
                
                else:
                    model_dict['image_decoder'] = MyDecoder(input_channel=512 // reduce_factor, up_type='Conv2', output_channel=self.image_ch,
                                                feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d, last_act=image_decoder_last_act)

            shape_encoder,shape_decoder =self.build_shape_encoder_decoder(network_type)
            if shape_decoder is not None:
                model_dict['shape_decoder'] =  shape_decoder
            if shape_encoder is not None:
                model_dict['shape_encoder'] =  shape_encoder
            
        elif 'Unet' in network_type:
            if '16' in network_type: 
                reduce_factor = 4
            elif '64' in network_type: 
                reduce_factor = 1
            else: raise NotImplementedError
            hidden_size = 768 ## transformer hidden size

            if 'enable_code_filter' in network_type:
                enable_code_filter=True
            else: enable_code_filter=False
            
            if 'UnetTransformer' in network_type:
                print ('use UnetTransformer for image segmentation,enable_code_filter:',enable_code_filter)
                image_encoder = UNETR_Encoder(in_channels=self.image_ch,img_size=self.image_size, feature_size=64//reduce_factor, hidden_size=hidden_size ,norm_name='batch', spatial_dims=2,enable_code_filter=enable_code_filter)
                segmentation_decoder = UNETR_Decoder(out_channels=self.num_classes,feature_size=64//reduce_factor,hidden_size=hidden_size, norm_name='batch', spatial_dims=2)
            else:
                if 'leaky_relu' in network_type:
                    act = nn.LeakyReLU
                else: act = nn.ReLU
                print ('use Unet for image segmentation, activation:',act)
                image_encoder = UnetEncoder(input_channel=self.image_ch, reduce_factor= reduce_factor,encoder_dropout=self.encoder_dropout,norm=nn.BatchNorm2d, enable_code_filter=enable_code_filter,activation=act)
                segmentation_decoder = UnetDecoder(n_classes=self.num_classes, reduce_factor=reduce_factor,decoder_dropout=self.decoder_dropout,norm=nn.BatchNorm2d,activation=act, last_act=None)
            model_dict = {'image_encoder': image_encoder,
                    'segmentation_decoder': segmentation_decoder,
                    }
            if not 'no_im_recon' in network_type:               
                if not 'Unet_im_recon' in network_type:
                        print ('use standard decoder for image generation')
                        model_dict['image_decoder'] = MyDecoder(input_channel=512 // reduce_factor, up_type='Conv2', output_channel=self.image_ch,
                                        feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d, last_act=image_decoder_last_act)
                else:
                    if 'UnetTransformer' in network_type:
                        print ('use Unet transformer-based decoder for image generation')
                        model_dict['image_decoder'] =UNETR_Decoder(out_channels=self.image_ch,feature_size=64//reduce_factor,hidden_size=hidden_size, norm_name='batch', spatial_dims=2,last_act=image_decoder_last_act)
                    else: 
                        if 'leaky_relu' in network_type:
                            act = nn.LeakyReLU
                        else: act = nn.ReLU
                        print ('use standard Unet decoder for image generation,activation:',act)
                        ## use Unet style decoder TODO: check if it is beneficial to add a Unet-style image decoder.
                        model_dict['image_decoder'] = UnetDecoder(n_classes=self.image_ch, reduce_factor=reduce_factor,decoder_dropout=self.decoder_dropout,norm=nn.BatchNorm2d,up_type='Conv2',last_act=last_act,activation=image_decoder_last_act)
            shape_encoder,shape_decoder = self.build_shape_encoder_decoder(network_type)
            if shape_decoder is not None:
                model_dict['shape_decoder'] =  shape_decoder
            if shape_encoder is not None:
                model_dict['shape_encoder'] =  shape_encoder
            print('init {}'.format(network_type))

        else:
            print (f'no {network_type} found')
            raise NotImplementedError
        print ('initialize model')
        model_dict  =self.init_model_with_pretrained(model_dict, checkpoint_dir)
        print ('initialize model end')
        print ('convert to cuda model')
        if self.use_gpu:
            for name, module in model_dict.items():
                if module is not None and isinstance(module, nn.Module):
                    model_dict[name] = module.cuda()
        print ('convert to cuda model end')
        self.model = model_dict
        return model_dict

    def parameters(self):
        return itertools.chain([module.parameters() for module in self.model.values()])

    def named_parameters(self):
        return itertools.chain([module.named_parameters() for module in self.model.values()])

    def init_model(self, model, resume_path=None):
        if not resume_path is None:
            if not resume_path == '':
                if not os.path.exists(resume_path):
                    print('path: {} not exist'.format(resume_path))
                    return None
                try:
                    model.load_state_dict(torch.load(resume_path))
                    print(f'load saved params from {resume_path}')
                except:
                    try:
                        # dummy code for some historical reason.
                        model.load_state_dict(torch.load(resume_path)['model_state'], strict=False)
                    except:
                        print('fail to load checkpoint under {}'.format(resume_path))
            else:
                print('can not find checkpoint under {}'.format(resume_path))
        else:
            try:
                init_weights(model, init_type='kaiming')
                print('init network')
            except:
                print('failed to init model')
        return model

    def init_model_with_pretrained(self, model_dict, checkpoint_dir):
        # load weights for each module
        for module_name,module in model_dict.items():
            if checkpoint_dir is not None and not checkpoint_dir == "":
                path = join(checkpoint_dir, f'{module_name}.pth')
            else: path = None
            if module is not None:
                module = self.init_model(module, resume_path = path)
                model_dict[module_name] = module
        return model_dict
    
    def run(self, input, disable_track_bn_stats=False):
        (latent_code_i, latent_code_s), init_predict = self.fast_predict(
            input, disable_track_bn_stats=disable_track_bn_stats)
        self.z_i = latent_code_i
        self.z_s = latent_code_s
        recon_image =self.decoder_inference(decoder_name='image_decoder', latent_code=latent_code_i, disable_track_bn_stats=disable_track_bn_stats)
       
        if 'no_STN' in self.network_type:
            refined_predict = init_predict
        else:
            refined_predict = self.recon_shape(init_predict, image=image, is_label_map=False,recon_image=recon_image,
                                            disable_track_bn_stats=disable_track_bn_stats)

        return recon_image, init_predict, refined_predict

    def encode_image(self, input, domain_id=0, disable_track_bn_stats=False):
        # FTN encoders
        encoder = self.model['image_encoder']
        if disable_track_bn_stats:
            with _disable_tracking_bn_stats(encoder):
                if self.network_type.startswith('Unet'):
                    z = encoder(input)
                else: 
                    z= encoder.general_encoder(input, domain_id)
        else:
            if self.network_type.startswith('Unet'):
                z = encoder(input)
            else: 
                z= encoder.general_encoder(input, domain_id)
        latent_code_i, latent_code_s = self.filter_code(z, disable_track_bn_stats)
        return latent_code_i, latent_code_s

    def filter_code(self, z, disable_track_bn_stats):
        latent_code_i = z
        if  self.network_type.startswith('Unet'):
            if 'enable_code_filter' in self.network_type:
                if disable_track_bn_stats:
                    with _disable_tracking_bn_stats(self.model['image_encoder']):
                            latent_code_s = self.model['image_encoder'].filter_code(z)
                else: 
                    latent_code_s = self.model['image_encoder'].filter_code(z)
                if 'Unet_im_recon' in self.network_type:
                    latent_code_i = z
                else:
                    latent_code_i = z[-1]
            else:
                latent_code_s = z ## use the cascaded feature for image segmentation with skip connections
                
                if 'Unet_im_recon' in self.network_type:
                    latent_code_i = z
                else:
                    #print ('use the bottom feature for image reconstruction')
                    latent_code_i = z[-1] ## use the feature from the last conv layers as the latent code for image recon
        else:
            if 'w_o_filter' in self.network_type:
                latent_code_i = z
                latent_code_s = z
            else:
                latent_code_i = z
                if disable_track_bn_stats:
                    with _disable_tracking_bn_stats(self.model['image_encoder']):
                        latent_code_s = self.model['image_encoder'].filter_code(z)
                else:
                    latent_code_s = self.model['image_encoder'].filter_code(z)
                if 'share_code' in self.network_type:
                    # z_i and z_s are shared (after filter) ## for ablation study
                    latent_code_i = latent_code_s

        self.latent_code['image'] = latent_code_i
        self.latent_code['segmentation'] = latent_code_s
        return latent_code_i, latent_code_s

    def encode_shape(self, segmentation, is_label_map=False, image=None, disable_track_bn_stats=False, temperature=None):
        '''
        STN: encoder function: S -> latent_z (STN)
        given a logit from the network or gt labels, encode it to the latent space
        '''
        if temperature is None:
            temperature = self.temperature
        prediction_map = construct_input(segmentation, image=image, num_classes=self.num_classes, apply_softmax=not is_label_map,
                                         is_labelmap=is_label_map, temperature=temperature, use_gpu=self.use_gpu, smooth_label=False)
        shape_code = self.decoder_inference(decoder_name='shape_encoder', latent_code=prediction_map, disable_track_bn_stats=disable_track_bn_stats)
        self.latent_code['shape'] = shape_code
        if is_label_map:
            self.gt_z = shape_code
        return shape_code


    def recon_shape(self, segmentation_logit, is_label_map, image=None,recon_image=None,disable_track_bn_stats=False):
        '''
        STN: shape refinement/correction: S'-> STN(S)
        return logit of reconstructed shape
        '''
        if 'no_STN' in self.network_type:
            shape = segmentation_logit
        else:
            if self.separate_training:
                segmentation_logit = segmentation_logit.detach().clone()
             # shape recon loss
            if 'w_image' in self.network_type:
                assert image is not None
                image = image
            elif 'w_recon_image' in self.network_type:
                assert recon_image is not None
                image = recon_image
            elif 'w_dual_image' in self.network_type:
                assert recon_image is not None and image is not None
                image = torch.cat([image, recon_image], dim=1)
            else:
                image = None
            shape_code = self.encode_shape(segmentation=segmentation_logit, is_label_map=is_label_map, image=image,
                                                                disable_track_bn_stats=disable_track_bn_stats)
            shape = self.decoder_inference(decoder_name='shape_decoder', latent_code=shape_code, disable_track_bn_stats=disable_track_bn_stats)
        
        return shape

    def recon_image(self, image, domain_id=0, disable_track_bn_stats=False):
        '''
        FTN: image recon, I-> FTN-> I'
        return reconstructed shape
        '''
        z_i, z_s = self.encode_image(image, domain_id=domain_id, disable_track_bn_stats=disable_track_bn_stats)
        x = self.decoder_inference(decoder_name='image_decoder', latent_code=z_i, disable_track_bn_stats=disable_track_bn_stats)
        return x

    def forward(self, input, disable_track_bn_stats=False):
        '''
        predict fast segmentation (FTN)
        '''
        zs, predict = self.fast_predict(input, disable_track_bn_stats)
        return predict

    def eval(self):
        self.train(if_testing=True)

    def requires_grad_(self, requires_grad=True):
        for module in self.model.values:
            for p in module.parameters():
                p.requires_grad = requires_grad

    def get_modules(self):
        return self.model.values

    def generate_max_style_image(self, image_code, decoder_layers_indexes=[3,4,5], channel_num=[128, 64, 32, 16, 16,1],
                                              p=0.5,
                                              n_iter=5, mix_style=True, lr=0.1, no_noise=False,
                                              reference_image=None, reference_segmentation=None,
                                              noise_learnable=True,
                                              mix_learnable=True,
                                              loss_types=['seg'],
                                              loss_weights=[1],
                                              always_use_beta=False,debug=False):
        """_summary_
        MaxStyle: apply style mixing and noise perturbation to intermediate layers in the decoder, to get style augmented recon images
        support adversararial trainining to optimize the style compositional parameters: lambda, epsilon_gamma, epsilon_bet
        Args:
            image_code (tensor): 4-d tensor,latent feature code
            decoder_layers_indexes (list, optional): decoder_layers_indexes. Defaults to [].
            lmda (float, optional): float. if specified, will set a fixed value for mixing/extraplation. Defaults to None.
            mix (str, optional): types of style mixing: random, gaussian, worst, reverse. Defaults to 'random'.
            if_extraplolate (bool, optional): if true, will perform style extrapolation. Defaults to False.
            random_layers (bool, optional): if true, will find random list of intermediate features for style augmentation. Defaults to False.
        Returns:
            [type]: [description
        Args:
            image_code (tensor): 4-d tensor,latent feature code
            decoder_layers_indexes (list<int> optional): decoder_layers_indexes. Defaults to [3,4,5].
            channel_num (list<int>): the number of features in *each* conv block. Defaults to [128, 64, 32, 16, 16,1].
            p (float, optional): the probability of applying maxstyle. Defaults to 0.5.
            n_iter (int, optional): number of adversarial training for style optimization. Defaults to 5.
            mix_style (bool, optional): whether to apply style mixing. Defaults to True.
            lr (float, optional): the learning rate for style optimization. Defaults to 0.1.
            no_noise (bool, optional): disable noise perturbation. Defaults to False.
            reference_image (tensor, optional): original input image. Defaults to None.
            reference_segmentation (tensor, optional): reference segmentation to compute loss function. Defaults to None.
            noise_learnable (bool, optional): whether to make noise-related parameters learnable. Defaults to True.
            mix_learnable (bool, optional): whether to make style-mixing related parameters learnable. Defaults to True.
            loss_types (list<string>, optional): specified loss type names. Defaults to ['seg'].
            loss_weights (list<float>, optional): corresponding loss weights. Defaults to [1].
            always_use_beta (bool, optional): whether to sample style mixing parameters lamdba from a beta distribution. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.

        Raises:
            ValueError: _description_
        Returns:
            image tensor: style augmented images or recon images
        """
        recon_image = None
        if not len(decoder_layers_indexes) > 0:
           recon_image= self.decoder_inference(decoder_name='image_decoder', latent_code=image_code, disable_track_bn_stats=False)
        else:
            old_state = {}
            for name, module in self.model.items():
                old_state[name] = module.training
                set_grad(module, requires_grad=False)
            decoder_function = self.model['image_decoder']

            style_augmentor_dict = {}
            optimizer  = None
            nn_style_augmentor_dict = None
            # Set up MaxStyle layers
            if isinstance(image_code,list):
                batch_size = image_code[0].size(0)
            else:
                batch_size = image_code.size(0)
            for i in decoder_layers_indexes:
                module = MaxStyle(batch_size, channel_num[i], p=p, mix_style=mix_style,
                    no_noise=no_noise, mix_learnable=mix_learnable, noise_learnable=noise_learnable,
                    always_use_beta=always_use_beta,debug=debug)
                style_augmentor_dict[str(i)] = module
            nn_style_augmentor_dict = nn.ModuleDict(style_augmentor_dict)

            # Set up optimizer(s)
            optimize = True
            if n_iter > 0:
                if len(list(nn_style_augmentor_dict.parameters())) == 0:
                    optimize = False
                else:
                    assert reference_image is not None and reference_segmentation is not None, 'must provide reference images and segmentations'
                    self.zero_grad()
                    optimizer  = torch.optim.Adam(nn_style_augmentor_dict.parameters(), lr=lr)
           
            for i in range(n_iter + 1):
                self.zero_grad()
                nn_style_augmentor_dict.zero_grad()
                if i > 0:
                    ## compute loss 
                    if not optimize:
                        break                    
                    optimizer.zero_grad()
                    latent_code_i,latent_code_s = self.encode_image(recon_image, disable_track_bn_stats=True)

                    p = self.decoder_inference(decoder=self.model['segmentation_decoder'],
                                                                latent_code=latent_code_s, eval=False, disable_track_bn_stats=True)

                    loss = 0
                    for l_w, ltype in zip(loss_weights, loss_types):
                        if ltype == 'seg':
                            l = -basic_loss_fn(pred=p, target=reference_segmentation, loss_type='cross entropy', class_weights=self.class_weights, use_gpu=self.use_gpu)
                        else:
                            raise ValueError('loss type {} not supported'.format(ltype))
                        loss = loss + l_w * l
                    ## optimize style parameters
                    optimizer.zero_grad()
                    loss.backward(retain_graph=False) if not 'Unet_im_recon' in self.network_type else loss.backward(retain_graph=True)
                    optimizer.step()
                    nn_style_augmentor_dict.zero_grad()
                ## get style augmented images
                recon_image  = decoder_function.apply_max_style(image_code, 
                        decoder_layers_indexes=decoder_layers_indexes,nn_style_augmentor_dict=nn_style_augmentor_dict)
            for name, module in self.model.items():
                set_grad(module, requires_grad=old_state[name])
        torch.cuda.empty_cache()
        self.zero_grad()
        return recon_image.detach().clone()

    def perturb_latent_code(self, latent_code, decoder_function, label_y=None,
                            perturb_type='random', threshold=0.5,
                            if_soft=False, random_threshold=False,
                            loss_type='mse', image_y=None, if_detach=False):
        """

        Args:
            latent_code (torch tensor): latent code z (a low-dimensional latent representation)
            decoder_function (nn.module): decoder function. a function that maps the latent code to the output space (image/label)
            label_y (torch tensor, optional): target value. Defaults to None. For targeted masking, it requires a target to compute the loss for gradient computation.
            perturb_type (str, optional): Names of mask methods. Defaults to 'random'. If random, will randomly select a method from the pool: ['dropout', 'spatial', 'channel']
            threshold (float, optional): dropout rate for random dropout or threshold for targeted masking:  mask codes with top p% gradients. Defaults to 0.5.
            if_soft (bool, optional): Use annealing factor to produce a soft mask with mask values sampled from [0,0.5] instead of 0. Defaults to False.
            random_threshold (bool, optional): Random sample a threshold from (0,threshold]. Defaults to False.
            loss_type (str, optional): Task-specific loss for targeted masking. Defaults to 'mse'.
            if_detach: If set to ``True``, will return the cloned masked code. Defaults to False
        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        assert perturb_type in ['random', 'dropout', 'spatial', 'channel', 'RSC', 'no_dropout'], 'invalid method name'

        if perturb_type == 'random':
            # random select a perturb type from 'dropout', 'spatial', 'channel'
            random_perturb_candidates = ['dropout', 'spatial', 'channel']
            random.shuffle(random_perturb_candidates)
            perturb_type = random_perturb_candidates[0]

        if perturb_type == 'RSC' or perturb_type == 'no_dropout':
            # no random dropout
            random_perturb_candidates = ['spatial', 'channel']
            random.shuffle(random_perturb_candidates)
            perturb_type = random_perturb_candidates[0]

            # print(perturb_type)
        if perturb_type == 'dropout':
            masked_latent_code = F.dropout2d(latent_code, p=threshold)
            mask = torch.where(masked_latent_code == latent_code,
                               torch.ones_like(masked_latent_code),
                               torch.zeros_like(masked_latent_code))
        else:
            assert loss_type in ['mse', 'ce', 'corr', 'ce-mse', 'l1', 'l2'], 'not implemented loss'

            if perturb_type == 'spatial':
                masked_latent_code, mask = mask_latent_code_spatial_wise(latent_code, num_classes=self.num_classes, decoder_function=decoder_function,
                                                                         label=label_y, percentile=threshold, random=random_threshold, loss_type=loss_type, if_detach=if_detach, if_soft=if_soft)
            elif perturb_type == 'channel':
                masked_latent_code, mask = mask_latent_code_channel_wise(latent_code, num_classes=self.num_classes, decoder_function=decoder_function,
                                                                         label=label_y, percentile=threshold, random=random_threshold, loss_type=loss_type, if_detach=if_detach, if_soft=if_soft)
            else:
                raise NotImplementedError
        if if_detach:
            masked_latent_code = masked_latent_code.detach().clone()
        torch.cuda.empty_cache()
        return masked_latent_code, mask

    def predict(self, input, softmax=False, n_iter=None):
        if n_iter is None:
            n_iter = self.n_iter
        else:
            n_iter = n_iter
            gc.collect()  # collect garbage
        self.eval()
       
        with torch.inference_mode():
            if  'no_STN' in self.network_type or n_iter <= 1:
                (zi,zs), pred = self.fast_predict(input)
            elif n_iter == 2:
                recon_image, init_predict, pred = self.run(input)
            else:
                raise ValueError

        if softmax:
            pred = torch.softmax(pred, dim=1)
        torch.cuda.empty_cache()
        return pred

    def decoder_inference(self, latent_code, decoder_name="", decoder=None, eval=False, disable_track_bn_stats=False):
        if decoder is None:
            try:
                decoder = self.model[decoder_name]
            except:
                raise ValueError("error! no decoder named {}".format(decoder_name))
        assert decoder is not None, 'cannot find decoder' 
        decoder_state = decoder.training

        if eval:
            decoder.eval()
            with torch.inference_mode():
                logit = decoder(latent_code)

        else:
            if disable_track_bn_stats:
                with _disable_tracking_bn_stats(decoder):
                    logit = decoder(latent_code)

            else:
                logit = decoder(latent_code)

        decoder.train(mode=decoder_state)
        return logit

    def compute_image_recon_loss(self, input_image, target, rec_loss_type=None):
        if rec_loss_type is None:
            rec_loss_type = self.rec_loss_type
        if rec_loss_type == 'l2':
            loss = 0.5 * torch.nn.MSELoss(reduction='mean')(input_image, target)
        elif rec_loss_type == 'l1':
            loss = torch.nn.L1Loss()(input_image, target)
        elif rec_loss_type == 'ngf':
            loss = NGF_Loss()(input_image, target)
        else:
            raise NotImplementedError
        return loss

    def standard_training(self, clean_image_l, label_l, perturbed_image, compute_gt_recon=True, update_latent=True, if_latent_code_consistency=False, disable_track_bn_stats=False, domain_id=0, return_output=False):
        """
        compute standard training loss
        Args:
            clean_image_l (torch tensor): original images (w/o corruption) NCHW
            label_l (torch tensor): reference segmentation. NHW
            perturbed_image (torch tensor): corrupted/noisy images. NCHW
            separate_training (bool, optional): if true, will block the gradients flow from STN to FTN. Defaults to False.
            compute_gt_recon (bool, optional): compute shape correction loss where input to STN is the ground truth map. Defaults to True.
            update_latent (bool, optional): save the latent codes. Defaults to True.

        Returns:
            standard_supervised_loss (float tensor): task-specific loss (ce loss for segmentation)
            image_recon_loss (float tensor): image reconstruction loss (mse for image recon)
            gt_shape_recon_loss (float tensor): shape correction loss (reconstruct the input label map)
            pred_shape_recon_loss (float tensor): shape correction loss (refine the output from FTN)
        """
        self.train()
        zero = torch.tensor(0., device=clean_image_l.device)

        (z_i, z_s), y_0 = self.fast_predict(perturbed_image, domain_id=domain_id, disable_track_bn_stats=disable_track_bn_stats)
        if update_latent:
            self.z_i = z_i
            self.z_s = z_s
        # seg task loss
        seg_loss = cross_entropy_2D(y_0, label_l.detach(), weight=self.class_weights)

        # image recon loss
        if not "no_im_recon" in self.network_type:
            recon_image = self.decoder_inference(decoder_name = "image_decoder",latent_code = z_i,disable_track_bn_stats=disable_track_bn_stats)
            assert recon_image is not None, 'recon image is None'
            image_recon_loss = self.compute_image_recon_loss(recon_image, target=clean_image_l.detach())
        else:
            recon_image =None
            image_recon_loss = zero
        self.recon_image = recon_image

        if 'no_STN' in self.network_type:
            pred_shape_recon_loss, gt_shape_recon_loss = zero, zero
            p_recon=y_0
        else:
            if compute_gt_recon:
                gt_recon = self.recon_shape(label_l, image=perturbed_image, is_label_map=True, recon_image=recon_image,
                                            disable_track_bn_stats=disable_track_bn_stats)
                gt_shape_recon_loss = cross_entropy_2D(gt_recon, label_l.detach(), weight=self.class_weights)
            else:
                gt_shape_recon_loss = zero

            p_recon = self.recon_shape(y_0, image=perturbed_image, is_label_map=False,recon_image=recon_image,
                                    disable_track_bn_stats=disable_track_bn_stats)
            pred_shape_recon_loss = cross_entropy_2D(p_recon, label_l.detach(), weight=self.class_weights)
        if return_output:
            return seg_loss, image_recon_loss, gt_shape_recon_loss, pred_shape_recon_loss, recon_image,y_0,p_recon

        else:
            return seg_loss, image_recon_loss, gt_shape_recon_loss, pred_shape_recon_loss

    def hard_example_generation(self,
                                clean_image_l,
                                label_l,
                                z_i=None,
                                z_s=None,
                                gen_corrupted_seg=True,
                                gen_corrupted_image=True,
                                corrupted_image_DA_config={"loss_name": "mse",
                                                           "mask_type": "random",
                                                           "max_threshold": 0.5,
                                                           "random_threshold": True,
                                                           "if_soft": True},
                                corrupted_seg_DA_config={"loss_name": "ce",
                                                         "mask_type": "random",
                                                         "max_threshold": 0.5,
                                                         "random_threshold": True,
                                                         "random_threshold": True,
                                                         "if_soft": True}):
        # fixed segmentation decoder, we perturb the latent space to get corrupted segmentation, and use them to train our denoising shape autoencodeer,
        perturbed_image_0, perturbed_y_0 = None, None
        torch.cuda.empty_cache()
        if z_i is None:
            z_i = self.z_i
        if z_s is None:
            z_s = self.z_s
        if gen_corrupted_image:
            self.reset_all_optimizers()
            perturbed_z_i_0, img_code_mask = self.perturb_latent_code(latent_code=z_i,
                                                                      label_y=clean_image_l,
                                                                      perturb_type=corrupted_image_DA_config["mask_type"],
                                                                      decoder_function=self.model['image_decoder'],
                                                                      loss_type=corrupted_image_DA_config["loss_name"],
                                                                      threshold=corrupted_image_DA_config["max_threshold"],
                                                                      random_threshold=corrupted_image_DA_config["random_threshold"],
                                                                      if_detach=True, if_soft=corrupted_image_DA_config["if_soft"])
            perturbed_image_0 = self.decoder_inference(decoder=self.model['image_decoder'],
                                                       latent_code=perturbed_z_i_0, eval=False, disable_track_bn_stats=True)
            perturbed_image_0 = perturbed_image_0.detach().clone()
            perturbed_image_0 = rescale_intensity(perturbed_image_0, 0, 1)

        if gen_corrupted_seg:
            self.reset_all_optimizers()
            # print ('perform shape code perturbation')
            perturbed_z_0, shape_code_mask = self.perturb_latent_code(latent_code=z_s,
                                                                      label_y=label_l,
                                                                      perturb_type=corrupted_seg_DA_config["mask_type"],
                                                                      decoder_function=self.model['segmentation_decoder'],
                                                                      loss_type=corrupted_seg_DA_config["loss_name"],
                                                                      threshold=corrupted_seg_DA_config["max_threshold"],
                                                                      random_threshold=corrupted_seg_DA_config["random_threshold"],
                                                                      if_detach=True, if_soft=corrupted_seg_DA_config["if_soft"])

            perturbed_y_0 = self.decoder_inference(decoder=self.model['segmentation_decoder'],
                                                   latent_code=perturbed_z_0, eval=False, disable_track_bn_stats=True)
        return perturbed_image_0, perturbed_y_0

    def hard_example_traininng(self, perturbed_image, clean_image_l, perturbed_seg, label_l, use_gpu=True, if_latent_code_consistency=False, standard_input_image=None, standard_recon_image=None):
        """
        compute hard training loss
        Args:
           perturbed_image (torch tensor): corrupted/noisy images. NCHW
           clean_image_l (torch tensor): original images (w/o corruption) NCHW
           perturbed_seg (torch tensor): corrupted segmentation. NCHW
           label_l (torch tensor): reference segmentation. NHW
           use gpu (bool, optional): use gpu. Defaults to True.
        Returns:
           seg_loss (float tensor):  segmentation loss given the corrupted image
           recon_loss (float tensor): image reconstruction loss (input is the corrupted imaeg)
           shape_loss (float tensor): shape correction loss (input is the FTN's prediction on corrupted images)
           perturbed_p_recon_loss (float tensor): shape correction loss (input is the generated corrupted segmentations by code masking)
        """
        zero = torch.tensor(0., device=clean_image_l.device)
        seg_loss, recon_loss, shape_loss, perturbed_p_recon_loss = zero, zero, zero, zero
        if 'DS_FCN_' in self.network_type:
            disable_track_bn_stats = False
            domain_id = 1
        else:
            disable_track_bn_stats = True
            domain_id = 0

        if perturbed_image is not None:
            # w. corrupted image
            # perturbed_image = makeVariable(perturbed_image.detach().clone(), use_gpu=use_gpu, type='float')
            seg_loss, recon_loss, _, shape_loss = self.standard_training(clean_image_l=clean_image_l, label_l=label_l,
                                                                         perturbed_image=perturbed_image, compute_gt_recon=False, update_latent=False,
                                                                         disable_track_bn_stats=disable_track_bn_stats, domain_id=domain_id)

        if not 'no_STN' in self.network_type:
            if perturbed_seg is not None:
                perturbed_p_recon = self.recon_shape(perturbed_seg, image=standard_input_image,recon_image=standard_recon_image,
                                                    is_label_map=False, disable_track_bn_stats=disable_track_bn_stats)
                perturbed_p_recon_loss = basic_loss_fn(
                    pred=perturbed_p_recon, target=label_l, loss_type='cross entropy')
            else:perturbed_p_recon_loss = 0*seg_loss
        else:
            perturbed_p_recon_loss = 0*seg_loss

        return seg_loss, recon_loss, shape_loss, perturbed_p_recon_loss

    def fast_predict(self, input, domain_id=0, disable_track_bn_stats=False):
        """
        given an input image, return its latent code and pixelwise prediction

        Args:
            input ([type]): torch tensor
            disable_track_bn_stats (bool, optional):disable bn stats tracking. Defaults to False.

        Returns:
            z0: latent code tuple
            p0: pixelwise logits from the model
        """
        gc.collect()  # collect garbage
        if not self.training:
            with torch.inference_mode():
                z_i, z_s = self.encode_image(input, domain_id=domain_id, disable_track_bn_stats=disable_track_bn_stats)
                y_0 = self.decoder_inference(decoder=self.model['segmentation_decoder'], latent_code=z_s, disable_track_bn_stats=disable_track_bn_stats) 

        else:
            z_i, z_s = self.encode_image(input, domain_id, disable_track_bn_stats=disable_track_bn_stats)
            y_0 =self.decoder_inference(decoder=self.model['segmentation_decoder'], latent_code=z_s, disable_track_bn_stats=disable_track_bn_stats) 
        return (z_i, z_s), y_0

    def predict_w_reconstructed_image(self, image, domain_id=0):
        image_recon = self.recon_image(image, domain_id=domain_id)
        assert image_recon is not None, 'recon image is none'
        (zi, zs), pred = self.fast_predict(image_recon)
        return pred

    def evaluate(self, input, targets_npy, n_iter=None):
        '''
        evaluate the model performance

        :param input: 4-d tensor input: NCHW
        :param targets_npy: numpy ndarray: N*H*W
        :param running_metric: runnning metric for evaluatation
        :return:
        '''
        if n_iter is None:
            n_iter = self.n_iter
        gc.collect()  # collect garbage
        self.train(if_testing=True)
        pred = self.predict(input, n_iter=n_iter)
        pred_npy = pred.max(1)[1].cpu().numpy()
        self.running_metric.update(label_trues=targets_npy, label_preds=pred_npy)
        self.cur_eval_images = input.data.cpu().numpy()[:, 0, :, :]
        del input
        self.cur_eval_predicts = pred_npy
        self.cur_eval_gts = targets_npy  # N*H*W
        return pred

    def save_model(self, save_dir, epoch_iter, model_prefix=None, save_optimizers=False):
        if model_prefix is None:
            model_prefix = self.network_type
        epoch_path = join(save_dir, *[str(epoch_iter), 'checkpoints'])
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        for model_name, model in self.model.items():
            torch.save(model.state_dict(),
                       join(epoch_path, '{}.pth'.format(model_name)))
        if save_optimizers:
            for model_name, optimizer in self.optimizers.items():
                torch.save(optimizer.state_dict(),
                           join(epoch_path, '{}_optim.pth'.format(model_name)))

    def get_model_states_dict(self):
        model_states_dict = {}
        for model_name, model in self.model.items():
            state_dict = model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else model.state_dict()
            model_states_dict[model_name] = state_dict
        return model_states_dict

    def restore_model(self, model_state_dict):
        for model_name, model_dict in model_state_dict.items():
            self.model[model_name].load_state_dict(model_dict)

    def save_snapshots(self, save_dir, epoch, model_prefix='interrupted'):
        epoch_path = join(save_dir, *['interrupted', 'checkpoints'])
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        if model_prefix is None:
            model_prefix = self.network_type
        else:
            model_prefix = self.network_type+model_prefix
        save_path = join(epoch_path, model_prefix+ f'_{epoch}.pkl')
        model_states_dict = self.get_model_states_dict()
        optimizers_dict = {}
        for optimizer_name, optimizer in self.optimizers.items():
            optimizers_dict[optimizer_name] = optimizer.state_dict()
        state = {'network_type': self.network_type,
                 'epoch': epoch,
                 'model_state': model_states_dict,
                 'optimizer_state': optimizers_dict
                 }
        torch.save(state, save_path)
        return save_path

    def load_snapshots(self, file_path):
        """
        load checkpoints from the pkl file
        Args:
            file_path (str): path-to-checkpoint.pkl

        Returns:
            the epoch when saved (int):
        """
        start_epoch = 0
        if file_path is None:
            return start_epoch
        if file_path == '' or (not os.path.exists(file_path)):
            print(f'warning: {file_path} does not exists')
            return start_epoch
        try:
            checkpoint = torch.load(file_path)
        except:
            print('error in opening {}'.format(file_path))
        try:
            if self.model is None:
                self.model = self.get_network(network_type=checkpoint['network_type'])
            state_dicts = checkpoint['model_state']
            optimizer_states = checkpoint['optimizer_state']
            assert self.model, 'must initialize model first'
            assert self.optimizers, 'must initialize optimizer first'
            for k, v in self.model.items():
                v.load_state_dict(state_dicts[k])
            for k, v in self.optimizers.items():
                v.load_state_dict(optimizer_states[k])
            start_epoch = checkpoint['epoch']
            print("Loaded checkpoint '{}' (epoch {})".format(file_path, checkpoint['epoch']))
        except Exception as e:
            print('error: {} in loading {}'.format(e, file_path))
        return start_epoch

    def train(self, if_testing=False):
        self.training = True
        assert self.model, 'no model exists'
        for k, v in self.model.items():
            if not if_testing:
                if v is None: Warning('sub model {} is None'.format(k))
                else:
                    v.train()
                    v.training = True
                    set_grad(v, requires_grad=True)
            else:
                if v is None: Warning('sub model {} is None'.format(k))
                else:
                    v.training = False
                    v.eval()

    def eval(self):
        self.training = False
        self.train(if_testing=True)

    def reset_all_optimizers(self):
        if self.optimizers is None:
            self.set_optimizers()
        for k, v in self.optimizers.items():
            v.zero_grad()

    def zero_grad(self):
        for k, network in self.model.items():
            network.zero_grad()

    def get_optimizer(self, model_name=None):
        assert self.optimizers, 'please set optimizers first before fetching'
        if model_name is None:
            return self.optimizers
        else:
            return self.optimizers[model_name]

    def set_optimizers(self):
        assert self.model
        optimizers_dict = {}
        for model_name, model in self.model.items():
            print(f'set optimizer {self.optimizer_type} for: {model_name}')
            if self.optimizer_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.99)
            else: raise NotImplementedError
            optimizers_dict[model_name] = optimizer
        self.optimizers = optimizers_dict

    def set_schedulers(self, optimizer_dict):
        schedulers_dict= {}
        if self.optimizer_type == 'SGD':
            for k, v in optimizer_dict.items():
                schedulers_dict[k] = get_scheduler(v, lr_policy='step', lr_decay_iters=5)
        else:
            print ("No schedulers for optimizers")
        self.schedulers_dict = schedulers_dict

    def update_schedule(self):
        for k, v in self.schedulers_dict.items():
            v.step()

    def optimize_all_params(self):
        for k, v in self.optimizers.items():
            v.step()

    def optimize_params(self, model_name):
        self.optimizers[model_name].step()

    def reset_optimizer(self, model_name):
        self.optimizers[model_name].zero_grad()

    def set_running_metric(self):
        running_metric = runningScore(n_classes=self.num_classes)
        return running_metric

    def save_testing_images_results(self, save_dir, epoch_iter, max_slices=10, file_name='Seg_plots.png'):
        gts = self.cur_eval_gts
        predicts = self.cur_eval_predicts
        images = self.cur_eval_images
        save_testing_images_results(images, gts, predicts, save_dir, epoch_iter, max_slices, file_name)

if __name__ == '__main__':

    solver = AdvancedTripletReconSegmentationModel(
        network_type='FCN_16_standard', num_classes=4, n_iter=3, use_gpu=True)
    model = solver.model
    images = torch.randn(2, 1, 224, 224).to('cuda')
    pred = solver.predict(images)
    # print ('output',pred.size())
    # print ('latent',solver.latent_code)
