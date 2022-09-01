
from __future__ import print_function
import sys
import os
import argparse
import shutil
from os.path import join, exists
import gc
import random
from matplotlib import style
import numpy as np
import torch
import torchio as tio
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append('.')  # noqa

from src.common_utils.load_args import Params,get_value_from_dict
from src.common_utils.metrics import print_metric
from src.common_utils.basic_operations import check_dir, link_dir, delete_dir, rescale_intensity, set_seed
from src.dataset_loader.cardiac_general_dataset import Cardiac_General_Dataset
from src.dataset_loader.cardiac_ACDC_dataset import DATASET_NAME, CardiacACDCDataset
from src.dataset_loader.prostate_Decathlon_dataset import ProstateDataset
from src.dataset_loader.base_segmentation_dataset import ConcatDataSet
from src.dataset_loader.transform import Transformations  #

from src.models.model_util import makeVariable
from src.models.advanced_triplet_recon_segmentation_model import AdvancedTripletReconSegmentationModel
from src.models.custom_loss import VGGPerceptualLoss, basic_loss_fn, entropy_loss, kl_divergence, cross_entropy_2D
from src.advanced.rand_conv_aug import RandConvAug
from src.advanced.random_window_masking import random_inpainting, random_outpainting
from src.test_ACDC_triplet_segmentation import evaluate as cardiac_evaluate
from src.test_prostate_segmentation import evaluate as prostate_evaluate

import sys
sys.path.append('./advchain')  # noqa
from advchain.augmentor import ComposeAdversarialTransformSolver, AdvBias, AdvNoise


def seed_worker(worker_id=0):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sample_batch(dataiter, dataloader, SSL_flag=False):
    try:
        batch = next(dataiter)
    except StopIteration:
        dataiter = dataloader.__iter__()
        batch = next(dataiter)

    if SSL_flag:
        labelled_batch, unlabelled_batch = batch
    else:
        labelled_batch = batch
        unlabelled_batch = None
    return labelled_batch, unlabelled_batch, dataiter

def get_image_label(labelled_batch,keep_origin=True,use_gpu=True):
    image_l, label_l = labelled_batch['image'], labelled_batch['label']
    if keep_origin:
        image_orig, gt_orig = labelled_batch['origin_image'], labelled_batch['origin_label']
        image_l = torch.cat([image_l, image_orig], dim=0)
        label_l = torch.cat([label_l, gt_orig], dim=0)
    image_l = makeVariable(image_l, type='float', use_gpu=use_gpu, requires_grad=False)
    label_l = makeVariable(label_l, type='long', use_gpu=use_gpu, requires_grad=False)
    return image_l, label_l

def eval_model(experiment_name, segmentation_model, validate_loader):
    segmentation_model.running_metric.reset()
    segmentation_model.eval()
    for b_iter, test_batch in enumerate(tqdm(validate_loader)):
        clean_image_l, label_l = get_image_label(test_batch,keep_origin=False,use_gpu=use_gpu)
        random_sax_image_V = makeVariable(clean_image_l, type='float',
                                          use_gpu=use_gpu, requires_grad=True)
        segmentation_model.evaluate(input=random_sax_image_V,
                                    targets_npy=label_l.cpu().numpy(), n_iter=2)
    score = print_metric(segmentation_model.running_metric, name=experiment_name)
    # keep the best model
    curr_score = score['Mean IoU : \t']
    curr_acc = score['Mean Acc : \t']
    return curr_score, curr_acc


def train_network(experiment_name, dataset,
                  segmentation_solver,
                  experiment_opt,
                  log=False,
                  debug=False):
    '''

    :param experiment_name:
    :param dataset:

    :param resume_path:
    :param log:
    :return:
    '''
    # output setting
    global start_epoch, last_epoch, training_opt, crop_size, model_dir, log_dir, dataset_name

    # =========================dataset config==================================================#
    train_set = dataset[0]
    validate_set = dataset[1]
    batch_size = experiment_opt['learning']["batch_size"]
    keep_origin = experiment_opt['data']['keep_orig_image_label_pair_for_training']
    if keep_origin:
        train_batch_size = batch_size // 2
    else:
        train_batch_size = batch_size

    g = torch.Generator()
    if training_opt.seed is not None:
        g.manual_seed(training_opt.seed)
    train_loader = DataLoader(dataset=train_set, num_workers=training_opt.n_workers, batch_size=int(train_batch_size), shuffle=True, drop_last=True,
                              pin_memory=not training_opt.no_pin_memory, worker_init_fn=seed_worker, generator=g)
    validate_loader = DataLoader(dataset=validate_set, num_workers=training_opt.n_workers, batch_size=int(batch_size), shuffle=False,
                                 drop_last=False, pin_memory=not training_opt.no_pin_memory, worker_init_fn=seed_worker, generator=g)

   
    best_score = -10000

    if log:
        writer = SummaryWriter(log_dir=log_dir, purge_step=start_epoch)

   ##########advanced DA and cooperative training config############
    latent_DA = get_value_from_dict(experiment_opt['learning'],"latent_DA",False)
    max_style = get_value_from_dict(experiment_opt['learning'],"max_style",False)
    rand_conv = get_value_from_dict(experiment_opt['learning'],"rand_conv",False)
    RSC = get_value_from_dict(experiment_opt['learning'],"RSC",False)
    mix_style = get_value_from_dict(experiment_opt['learning'],"mix_style",False)
    DSU = get_value_from_dict(experiment_opt['learning'],"DSU",False)
    adv_noise = get_value_from_dict(experiment_opt['learning'],"adv_noise",False)
    adv_bias = get_value_from_dict(experiment_opt['learning'],"adv_bias",False)

    # =========================<<<<<start training>>>>>>>>=============================>
    stop_flag = False
    score_list = []

    segmentation_solver.reset_all_optimizers()
    segmentation_solver.train()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    g_count = 0
    total_loss = 0.
    loss_keys = ['loss/standard/total', 'loss/standard/seg', 'loss/standard/image', 'loss/standard/shape', 'loss/standard/gt_shape',
                'loss/hard/total', 'loss/hard/seg', 'loss/hard/image', 'loss/hard/shape','loss/hard/rand_conv', 'loss/hard/RSC','loss/hard/mix_style', 'loss/hard/DSU', 'loss/hard/adv_noise', 'loss/hard/adv_bias']
    loss_dict = {}
    for key in loss_keys:
        loss_dict[key] = torch.tensor(0., device=device)
    try:
        for i_epoch in range(start_epoch, experiment_opt['learning']['n_epochs']):
            last_epoch = i_epoch
            torch.cuda.empty_cache()
            gc.collect()  # collect garbage
      
            print('start training epoch {}'.format(i_epoch))
            i_iter =0
            if stop_flag:
                break
            for i_iter, sampled_batch in enumerate(tqdm(train_loader)):
                ######standard training############
                if training_opt.debug: 
                    if i_iter > 20: break
                gc.collect()  # collect garbage
                # step 1: get a batch of labelled images to get initial estimate
                segmentation_solver.train()
                segmentation_solver.reset_all_optimizers()

                clean_image_l,label_l = get_image_label(sampled_batch,keep_origin=keep_origin,use_gpu=use_gpu)
                batch_4d_size = clean_image_l.size()
                
                # add noise to input to train the FTN (same as training a standard denoising autoencoder to avoid overfitting)
                noise = 0.05 * torch.randn(batch_4d_size[0], batch_4d_size[1], batch_4d_size[2],
                                           batch_4d_size[3], device=clean_image_l.device, dtype=clean_image_l.dtype)
                image_l =  clean_image_l + noise
                if intensity_norm_type=='min_max':image_l = torch.clamp(image_l,clean_image_l.min() , clean_image_l.max())
                elif intensity_norm_type=='z_score':image_l = F.instance_norm(image_l, eps=1e-05, momentum=0.1, weight=None, bias=None)
                else: raise ValueError('intensity_norm_type not supported')
                image_l = makeVariable(image_l, use_gpu=use_gpu,requires_grad=False, type='float')

                # step 2: standard training
                seg_loss, image_recon_loss, gt_recon_loss, shape_recon_loss, easy_recon_image, p0, p_refine = segmentation_solver.standard_training(
                    clean_image_l, label_l, perturbed_image=image_l, return_output=True)
                ## get latent code for further latent space data augmentation
                z_i = segmentation_solver.z_i
                z_s = segmentation_solver.z_s
                standard_loss = seg_loss + image_recon_loss + shape_recon_loss + gt_recon_loss
                loss_dict['loss/standard/total'] += standard_loss.item()
                loss_dict['loss/standard/seg'] += seg_loss.item()
                loss_dict['loss/standard/image'] += image_recon_loss.item()
                loss_dict['loss/standard/shape'] += shape_recon_loss.item()
                loss_dict['loss/standard/gt_shape'] += gt_recon_loss.item()

                if latent_DA:
                    # MICCAI 2021 paper: latent space masking-based
                    # load_parameters:
                    # print('latent code masking configurations')
                    latentDA_config = experiment_opt['latent_DA']
                    if 'image code' in latentDA_config["mask_scope"]:
                        gen_corrupted_image = True
                        corrupted_image_DA_config = latentDA_config["image code"]
                        # print(latentDA_config["image code"])
                    else:
                        corrupted_image_DA_config = None
                        gen_corrupted_image = False
                    if 'shape code' in latentDA_config["mask_scope"]:
                        gen_corrupted_seg = True
                        # print(latentDA_config["shape code"])
                        corrupted_seg_DA_config = latentDA_config["shape code"]
                    else:
                        corrupted_seg_DA_config = None
                        gen_corrupted_seg = False 
                    segmentation_solver.reset_all_optimizers()

                    perturbed_image_0, perturbed_y_0 = segmentation_solver.hard_example_generation(clean_image_l.detach().clone(),
                                                                                                   label_l.detach().clone(),
                                                                                                   z_i=z_i,
                                                                                                   z_s=z_s,
                                                                                                   gen_corrupted_seg=gen_corrupted_seg,
                                                                                                   gen_corrupted_image=gen_corrupted_image,
                                                                                                   corrupted_image_DA_config=corrupted_image_DA_config,
                                                                                                   corrupted_seg_DA_config=corrupted_seg_DA_config,
                                                                                                   )

                    seg_supervised_loss, corrupted_image_recon_loss, shape_recon_loss_2, corrupted_shape_recon_loss = segmentation_solver.hard_example_traininng(perturbed_image=perturbed_image_0,
                                                                                                                                                                 perturbed_seg=perturbed_y_0,
                                                                                                                                                                 clean_image_l=clean_image_l, label_l=label_l,
                                                                                                                                                                 standard_input_image=image_l.detach().clone(), standard_recon_image=easy_recon_image)

            

                    LDA_loss = seg_supervised_loss + corrupted_image_recon_loss + \
                        shape_recon_loss_2 + corrupted_shape_recon_loss
                    loss_dict['loss/hard/total'] += LDA_loss.item()
                    loss_dict['loss/hard/seg'] += seg_supervised_loss.item()
                    loss_dict['loss/hard/image'] += corrupted_image_recon_loss.item()
                    loss_dict['loss/hard/shape'] += (shape_recon_loss_2 +
                                                     corrupted_shape_recon_loss).item()
                    torch.cuda.empty_cache()
                    
                else:
                    LDA_loss = torch.tensor(0., device=device)
                
                if max_style:
                    # MICCAI 2022 paper: MaxStyle: feature style space augmentation
                    segmentation_solver.reset_all_optimizers()
                    always_use_beta =get_value_from_dict(experiment_opt['max_style'],"always_use_beta",False)
                    if '16' in segmentation_solver.network_type:
                        channel_num  = [128, 64, 32, 16, 16,1]
                    elif '64' in segmentation_solver.network_type:
                        channel_num  = [512, 256, 128, 64, 64,1]
                    else:
                        raise ValueError('network_type not supported')
                    stylized_images = segmentation_solver.generate_max_style_image(image_code=z_i,
                                                                                                channel_num=channel_num,
                                                                                                p=0.5,
                                                                                                decoder_layers_indexes=experiment_opt[
                                                                                                    'max_style']['decoder_layers_indexes'],
                                                                                                n_iter=experiment_opt['max_style']['n_iter'],
                                                                                                mix_style=experiment_opt['max_style']['mix_style'],
                                                                                                lr=experiment_opt['max_style']['lr'],
                                                                                                no_noise=experiment_opt['max_style']['no_noise'],
                                                                                                reference_image=clean_image_l,
                                                                                                reference_segmentation=label_l,
                                                                                                noise_learnable=experiment_opt['max_style']['noise_learnable'],
                                                                                                mix_learnable=experiment_opt['max_style']['mix_learnable'],
                                                                                                loss_types=experiment_opt['max_style']['loss_types'],
                                                                                                loss_weights=experiment_opt['max_style']['loss_weights'],
                                                                                                always_use_beta=always_use_beta,
                                                                                               )
                    stylized_images = stylized_images.detach().clone()
                    l_seg_1, l_rec, l_shape_1, l_shape_2 = segmentation_solver.hard_example_traininng(perturbed_image=stylized_images, perturbed_seg=None, clean_image_l=clean_image_l, label_l=label_l,
                                                                                standard_input_image=image_l.detach().clone(), standard_recon_image=easy_recon_image)
                    max_style_loss = l_rec + l_seg_1 + l_shape_1+l_shape_2
                    loss_dict['loss/hard/total'] += max_style_loss.item()
                    loss_dict['loss/hard/seg'] += l_seg_1.item()
                    loss_dict['loss/hard/image'] += l_rec.item()
                    loss_dict['loss/hard/shape'] += (l_shape_1+l_shape_2).item()
                else:
                    max_style_loss = torch.tensor(0., device=device)
                
                if rand_conv:
                    # apply randconv to input image three times
                    # [ICLR'21] Robust and Generalizable Visual Representation Learning via Random Convolutions
                    # https://openreview.net/pdf?id=BVSM0x3EDK6
                    recon_predictions, initial_predicts, final_predicts = [], [], []
                    for i in range(3):
                        aug_image = RandConvAug().transform(input_image=image_l)
                        recon_image, init_predict, refined_predict = segmentation_solver.run(aug_image,normalize_input=True)
                        recon_predictions.append(recon_image)
                        initial_predicts.append(F.softmax(init_predict, dim=1))
                        final_predicts.append(F.softmax(refined_predict, dim=1))
                    # average prediction
                    # average_image = sum(recon_predictions) / 3.0
                    # reference code: https://github.com/google-research/augmix/blob/master/cifar.py
                    average_FTN_log = torch.clamp(sum(initial_predicts) / 3.0, 1e-8, 1).log()
                    c = num_classes
                    average_FTN_log = average_FTN_log.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
               
                        # compute JS-like loss 1/3(KL(p1|p_mean)+ KL(p2|p_mean)+ KL(p3|p_mean))
                    rand_conv_loss = 0.
                    lamda = 10  # as suggested in the original paper
                    for rec, predict, stn_predict in zip(recon_predictions, initial_predicts, final_predicts):
                        l_rec = segmentation_solver.compute_image_recon_loss(rec, clean_image_l.detach())
                        # KL(P|P_mean)
                        predict = predict.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
                        l_seg_1 = lamda * F.kl_div(average_FTN_log, predict, reduction='batchmean')

                        if not 'no_STN' in segmentation_solver.network_type:
                            average_STN_log = torch.clamp(sum(final_predicts) / 3.0, 1e-8, 1).log()
                            average_STN_log = average_STN_log.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
                            stn_predict = stn_predict.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
                            l_shape_1 = lamda * F.kl_div(average_STN_log, stn_predict, reduction='batchmean')
                        else:
                            l_shape_1= 0

                        rand_conv_loss += l_rec + l_seg_1 + l_shape_1
                    rand_conv_loss /= 3.0
                    loss_dict['loss/hard/rand_conv'] += rand_conv_loss.item()
                else:
                    rand_conv_loss = torch.tensor(0., device=device)
                
                if RSC:
                    # RSC loss: use targeted hard masking for regularization
                    RSC_threshold = 1.0 / 3
                    perturbed_z_i_0, img_code_mask = segmentation_solver.perturb_latent_code(latent_code=z_i,
                                                                                             label_y=clean_image_l,
                                                                                             perturb_type='RSC',
                                                                                             decoder_function=segmentation_solver.model['image_decoder'],
                                                                                             loss_type='corr',  # make it consistent with original paper
                                                                                             threshold=RSC_threshold,
                                                                                             random_threshold=False,
                                                                                             if_detach=False, if_soft=False)
                    perturbed_z_s_0, shape_code_mask = segmentation_solver.perturb_latent_code(latent_code=z_s,
                                                                                               label_y=label_l,
                                                                                               perturb_type='RSC',
                                                                                               decoder_function=segmentation_solver.model[
                                                                                                   'segmentation_decoder'],
                                                                                               loss_type='corr',  # make it consistent with original paper
                                                                                               threshold=RSC_threshold,
                                                                                               random_threshold=False,
                                                                                               if_detach=False, if_soft=False)
                    ## force to predict seg with corrupted shape code
                    segmentation_logit = segmentation_solver.decoder_inference(decoder=segmentation_solver.model['segmentation_decoder'],
                                                                latent_code=z_s * shape_code_mask, eval=False, disable_track_bn_stats=True)

                    l_seg_2 = cross_entropy_2D(input=segmentation_logit,
                                               target=label_l.detach(), weight=segmentation_solver.class_weights)
                    
                    ## force to recon image and predict seg with corrupted image code
                    recon_image = segmentation_solver.decoder_inference(decoder= segmentation_solver.model['image_decoder'], latent_code = z_i * img_code_mask,eval=False, disable_track_bn_stats=True)
                    l_rec_reg = segmentation_solver.compute_image_recon_loss(recon_image, clean_image_l.detach().clone())

                    latent_code_i, new_z_s = segmentation_solver.filter_code(z_i * img_code_mask,disable_track_bn_stats=True)
                    segmentation_logit_1 = segmentation_solver.decoder_inference(decoder=segmentation_solver.model['segmentation_decoder'],
                                                                latent_code  = new_z_s, eval=False, disable_track_bn_stats=True)

                    l_seg_reg = cross_entropy_2D(segmentation_logit_1,
                                               target=label_l.detach(), weight=segmentation_solver.class_weights)
                  

                    if not 'no_STN' in segmentation_solver.network_type:
                        refined_predict = segmentation_solver.recon_shape(segmentation_logit, image=image_l, is_label_map=False,recon_image=easy_recon_image,disable_track_bn_stats=True)
                        l_shape_correct = cross_entropy_2D(refined_predict,
                                                            target=label_l.detach(), weight=segmentation_solver.class_weights)
                   
                        refined_segmentation_1 = segmentation_solver.recon_shape(
                            segmentation_logit_1, imgae=image_l,recon_image=recon_image,is_label_map=False, disable_track_bn_stats=True)

                        l_shape_correct_1 = cross_entropy_2D(refined_segmentation_1,
                                                 target=label_l.detach(), weight=segmentation_solver.class_weights)
                        l_shape_correct = (l_shape_correct + l_shape_correct_1) 
                    else: l_shape_correct = 0
                 
                    RSC_loss = l_rec_reg + l_seg_2+l_seg_reg + l_shape_correct
                    loss_dict['loss/hard/RSC'] += RSC_loss.item()

                else:
                    RSC_loss = torch.tensor(0., device=device)

                if mix_style or DSU:
                    ## perform feature style mixing or style perturbation with noise (DSU) for regularization
                    assert (mix_style and DSU) is False, 'mix_style and DSU cannot be True at the same time'
                    if mix_style:
                        layer_indexes = [1,2,3] ## best results with layer_indexes = [1,2,3]
                        mix='random'
                    else:
                        layer_indexes = [1,2,3,4,5,6] ## best results with layer_indexes = [1,2,3,4,5,6]
                        mix='gaussian'
                    aug_z_i, aug_z_s = segmentation_solver.generate_style_augmented_latent_code(
                        image=image_l, layers_indexes=layer_indexes, p=0.5, lmda=None, mix=mix)
                    recon_image = segmentation_solver.decoder_inference(decoder=segmentation_solver.model['image_decoder'],
                                                                latent_code  = aug_z_i, eval=False, disable_track_bn_stats=True)

                    segmentation_logit = segmentation_solver.decoder_inference(decoder=segmentation_solver.model['segmentation_decoder'],
                                                                latent_code  = aug_z_s, eval=False, disable_track_bn_stats=True)


                    l_seg = cross_entropy_2D(segmentation_logit, target=label_l.detach(),
                                             weight=segmentation_solver.class_weights)
                    if recon_image is not None:
                        l_rec = segmentation_solver.compute_image_recon_loss(
                            recon_image, clean_image_l.detach().clone())
                    else:
                        l_rec = 0 * l_seg
                  
                    if not 'no_STN' in segmentation_solver.network_type:
                        refined_predict = segmentation_solver.recon_shape(segmentation_logit, image=image_l, is_label_map=False,recon_image=easy_recon_image,disable_track_bn_stats=True)
                        l_shape_correct = cross_entropy_2D(refined_predict,
                                                            target=label_l.detach(), weight=segmentation_solver.class_weights)
                   
                    else: l_shape_correct = torch.tensor(0., device=device)
                    if mix_style: 
                        mix_style_loss = l_rec + l_seg + l_shape_correct
                        loss_dict['loss/hard/mix_style'] += mix_style_loss.item()
                        DSU_loss= torch.tensor(0., device=device)
                    else:
                        DSU_loss = l_rec + l_seg + l_shape_correct
                        loss_dict['loss/hard/DSU'] += DSU_loss.item()
                        mix_style_loss = torch.tensor(0., device=device)
                else:
                    mix_style_loss = torch.tensor(0., device=device)
                    DSU_loss = torch.tensor(0., device=device)
                    aug_z_s = None
                    aug_z_i = None
                
                if adv_noise:
                    ## perform adversarial noise for regularization
                    augmentor_function = AdvNoise(config_dict={'epsilon':0.1,
                                                                'xi': 1e-6,
                                                                'data_size': (clean_image_l.size(0), clean_image_l.size(1), clean_image_l.size(2), clean_image_l.size(3))},
                                                    debug=False)
                    divergence_types = ['kl']
                    divergence_weights = [1.0]
                    power_iteration = True

                    transformation_chain = [augmentor_function]
                    segmentation_solver.zero_grad()
                    segmentation_solver.eval()
                    adv_solver = ComposeAdversarialTransformSolver(
                        chain_of_transforms=transformation_chain,
                        divergence_types=divergence_types,
                        divergence_weights=divergence_weights,
                        use_gpu=True,
                        debug=False,
                        if_norm_image=True
                    )
                    consistency_loss = adv_solver.adversarial_training(
                        data=clean_image_l, model=segmentation_solver,
                        init_output = p0.detach().clone(),
                        n_iter=1,
                        lazy_load=[False],
                        optimize_flags=[True], power_iteration=True)
                    aug_image = adv_solver.adv_data.detach().clone()
                    torch.cuda.empty_cache()
                    segmentation_solver.train()
                    segmentation_solver.zero_grad()
                    seg_supervised_loss, corrupted_image_recon_loss, shape_recon_loss_2, corrupted_shape_recon_loss = segmentation_solver.hard_example_traininng(perturbed_image=aug_image,
                                                                                                                                                                 perturbed_seg=None,
                                                                                                                                                                 clean_image_l=clean_image_l, label_l=label_l,
                                                                                                                                                                 standard_input_image=image_l.detach().clone(), standard_recon_image=easy_recon_image)

            

                    adv_noise_loss = seg_supervised_loss + corrupted_image_recon_loss + \
                        shape_recon_loss_2 + corrupted_shape_recon_loss+ consistency_loss
                    loss_dict['loss/hard/adv_noise'] += adv_noise_loss.item()
                else:
                    adv_noise_loss = torch.tensor(0., device=device)

                if adv_bias:
                    ## MICCAI 2020: adversarial bias for regularization
                    if dataset_name == 'ACDC':
                        downscale = 2
                    else:
                        downscale = 4
                    augmentor_function = AdvBias(
                        config_dict={'epsilon': 0.4,
                                        'control_point_spacing':
                                        [clean_image_l.size(2) // 2, clean_image_l.size(3) // 2],
                                        'downscale': downscale,
                                        'data_size':
                                        (clean_image_l.size(0), clean_image_l.size(1),
                                        clean_image_l.size(2), clean_image_l.size(3)),
                                        'interpolation_order': 3,
                                        'init_mode': 'random',
                                        'space': 'log'}, debug=False)
                    divergence_types = ['kl', 'contour']
                    divergence_weights = [1.0, 0.5]
                    power_iteration = [False]

                    segmentation_solver.zero_grad()
                    segmentation_solver.eval()
                    adv_solver = ComposeAdversarialTransformSolver(
                        chain_of_transforms=[augmentor_function],
                        divergence_types=divergence_types,
                        divergence_weights=divergence_weights,
                        use_gpu=True,
                        debug=False,
                        if_norm_image=False
                    )
                    consistency_loss = adv_solver.adversarial_training(
                        data=clean_image_l, model=segmentation_solver,
                        init_output = p0.detach().clone(),
                        n_iter=1,
                        lazy_load=[False],
                        optimize_flags=[True], power_iteration=power_iteration)
                    aug_image = adv_solver.adv_data.detach().clone()
                    torch.cuda.empty_cache()
                    segmentation_solver.train()
                    segmentation_solver.zero_grad()
                    seg_supervised_loss, corrupted_image_recon_loss, shape_recon_loss_2, corrupted_shape_recon_loss = segmentation_solver.hard_example_traininng(perturbed_image=aug_image,
                                                                                                                                                                 perturbed_seg=None,
                                                                                                                                                                 clean_image_l=clean_image_l, label_l=label_l,
                                                                                                                                                                 standard_input_image=image_l.detach().clone(), standard_recon_image=easy_recon_image)

            

                    adv_bias_loss = seg_supervised_loss + corrupted_image_recon_loss + \
                        shape_recon_loss_2 + corrupted_shape_recon_loss+ consistency_loss
                    loss_dict['loss/hard/adv_bias'] += adv_bias_loss.item()
                else:
                    adv_bias_loss = torch.tensor(0., device=device)

                loss = standard_loss  + LDA_loss + max_style_loss+rand_conv_loss+RSC_loss+mix_style_loss+DSU_loss+adv_noise_loss+adv_bias_loss
                segmentation_solver.reset_all_optimizers()
                loss.backward()
                segmentation_solver.optimize_all_params()
                total_loss += loss.item()
                torch.cuda.empty_cache()
                if log:
                    for loss_name, loss_value in loss_dict.items():
                        writer.add_scalar(loss_name, (loss_value*1.0/(g_count+1)), g_count)
                g_count += 1                
                if i_iter > experiment_opt['learning']["max_iteration"]:
                    stop_flag = True
            print('{} network: {} epoch {} training loss iter: {}, total  loss: {}'.
                  format(experiment_name, experiment_opt['segmentation_model']["network_type"], i_epoch, i_iter, str(total_loss / (1.0 * g_count))))

            # =========================<<<<<start evaluating>>>>>>>>=============================
            curr_score, curr_acc = eval_model(experiment_name, segmentation_solver, validate_loader)
            score_list.append(curr_score)

            if log:
                writer.add_scalar('iou/val_iou', curr_score, i_epoch)
                writer.add_scalar('acc/val_acc', curr_acc, i_epoch)
            # save best models
            if best_score < curr_score:
                best_score = curr_score
                segmentation_solver.save_model(model_dir, epoch_iter='best',
                                               model_prefix=experiment_opt['segmentation_model']["network_type"])
                segmentation_solver.save_testing_images_results(model_dir, epoch_iter='best', max_slices=5)

            ###########save outputs ####################################################################
            if (i_epoch + 1) % experiment_opt["output"]["save_epoch_every_num_epochs"] == 0 or i_epoch == 0:
                segmentation_solver.save_model(model_dir, epoch_iter=i_epoch,
                                               model_prefix=experiment_opt['segmentation_model']["network_type"])
                segmentation_solver.save_testing_images_results(model_dir, epoch_iter=i_epoch, max_slices=5)
                gc.collect()  # collect garbage
            if segmentation_solver.schedulers_dict is not None:
                segmentation_solver.update_schedule()
            if stop_flag:
                break
            # avoid pytorch vram (GPU memory) usage keeps increasing
            torch.cuda.empty_cache()

        if log:
            try:
                writer.export_scalars_to_json(join(log_dir, experiment_name + ".json"))
                writer.close()
            except:
                print('already closed')
    except Exception as e:
        print('catch exception at epoch {}. error: {}'.format(str(i_epoch), e))
        if i_epoch > 0:
            segmentation_solver.save_snapshots(model_dir, epoch=i_epoch)
        last_epoch = i_epoch


# ========================= config==================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cooperative training and latent space DA for robust segmentation')
    # training config setting
    parser.add_argument("--json_config_path", type=str, default='./config/basic.json',
                        help='path of configurations')
    # data setting
    parser.add_argument("--dataset_name", type=str, default='ACDC',
                        help='dataset name')
    parser.add_argument("--cval", type=int, default=0,
                        help="cross validation subset")
    parser.add_argument("--data_setting", type=str, default="10",
                        help="data_setting:['one_shot','three_shot']")

    # model setting
    parser.add_argument("--resume_pkl_path", type=str, default=None, help='path-to-model-snapshot.pkl')
    parser.add_argument("--test_model_dir_path", type=str, default=None, help='directory-contains-model-pths')
    # output setting
    parser.add_argument("--save_dir", type=str,
                        default="./saved/",
                        help='path to resume the models')
    # visualizing the training/test performance
    parser.add_argument("--log", action='store_true', default=True,
                        help='use tensorboardX to tracking the training and testing curve')
    # advanced setting
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None,
                        help="set seed to reduce randomness in training")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="number of workers for data loaders")
    parser.add_argument("--no_pin_memory", action='store_true', default=False,
                        help='use pin memory for speed-up')
    parser.add_argument("--debug", action='store_true', default=False,
                        help='print information for debugging')
    parser.add_argument("--auto_test", action='store_true', default=True,
                        help='direct test after finishing training')
    parser.add_argument("--test_batch_size", type=int, default=25,
                        help='maximum batch size at test time')
    parser.add_argument("--no_train", action='store_true', default=False,
                        help='test only')
    parser.add_argument("--use_last_epoch", action='store_true', default=False,
                        help='use last epoch model for auto testing')

    # ========================= initialize training settings==================================================#
    # first load basic settings and then load args, finally load experiment configs
    training_opt = parser.parse_args()
    # torch.use_deterministic_algorithms(True)
    # limit randomness for reproducible research, ref: https://pytorch.org/docs/stable/notes/randomness.html
    # global setting
    set_seed(training_opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(training_opt.gpu)
    
    if training_opt.debug:
        import faulthandler
        faulthandler.enable()

    config_path = training_opt.json_config_path
    if exists(config_path):
        print('load params from {}'.format(config_path))
        experiment_opt = Params(config_path).dict
    else:  #
        print(config_path + 'does not not exists')
        raise FileNotFoundError

    # input dataset setting
    data_opt = experiment_opt['data']
    data_aug_policy_name = data_opt["data_aug_policy"]
    crop_size = data_opt['crop_size']
    intensity_norm_type = get_value_from_dict(data_opt,"intensity_norm_type","min_max")
    print ('intensity norm',intensity_norm_type)
    data_opt['new_spacing'] = get_value_from_dict(data_opt,"new_spacing",None)
    print ('new spacing',data_opt['new_spacing'])
    if not training_opt.no_train:
        # ========================= initialize training settings==================================================#
        tr = Transformations(data_aug_policy_name=data_opt["data_aug_policy"], pad_size=data_opt['pad_size'],
                            crop_size=data_opt['crop_size']).get_transformation()
        test_set = None
        dataset_name = data_opt['dataset_name']
        if 'ACDC' in dataset_name:
            frame = subset_name = data_opt['frame']
            if isinstance(frame, list):
                frame_list = frame
            else:
                frame_list = [frame]
            assert len(
                frame_list) <= 2, 'currently, only support concat two sets, please check your [frame] in the config file'
            train_set_list = []
            validate_set_list = []

            for frame in frame_list:
                train_set = CardiacACDCDataset(root_dir=data_opt["root_dir"], num_classes=data_opt["num_classes"],
                                            image_format_name=data_opt["image_format_name"],
                                            label_format_name=data_opt["label_format_name"],
                                            transform=tr['train'], subset_name=frame, split='train',
                                            data_setting_name=training_opt.data_setting,
                                            cval=training_opt.cval,
                                            keep_orig_image_label_pair=data_opt['keep_orig_image_label_pair_for_training'],
                                            use_cache=data_opt['use_cache'],
                                            myocardium_seg=data_opt['myocardium_only'],
                                            crop_size =  data_opt['crop_size'],
                                            new_spacing = data_opt['new_spacing'],
                                            right_ventricle_seg=data_opt['right_ventricle_only'],
                                            intensity_norm_type = intensity_norm_type

                                            )
                validate_set = CardiacACDCDataset(root_dir=data_opt["root_dir"], num_classes=data_opt["num_classes"],
                                                image_format_name=data_opt["image_format_name"],
                                                label_format_name=data_opt["label_format_name"],
                                                transform=tr['train'], subset_name=frame,
                                                split='validate',
                                                data_setting_name=training_opt.data_setting,
                                                cval=training_opt.cval,
                                                use_cache=data_opt['use_cache'],
                                                myocardium_seg=data_opt['myocardium_only'],
                                                crop_size =  data_opt['crop_size'],
                                                new_spacing = data_opt['new_spacing'],
                                                right_ventricle_seg=data_opt['right_ventricle_only'],
                                                keep_orig_image_label_pair=False,                                        
                                                intensity_norm_type = intensity_norm_type
                                                )
                train_set_list.append(train_set)
                validate_set_list.append(validate_set)

            if len(frame_list) > 1:
                train_set = torch.utils.data.ConcatDataset(datasets=train_set_list)
                validate_set = torch.utils.data.ConcatDataset(datasets=validate_set_list)
            else:
                train_set = train_set_list[0]
                validate_set = validate_set_list[0]
                del train_set_list
                del validate_set_list
        elif 'UKBB' in dataset_name:
            aframe = data_opt['frame']
            if isinstance(aframe, list):
                frame_list = aframe
            else:
                frame_list = [aframe]
            assert len(
                frame_list) <= 2, 'currently, only support concat two sets, please check your [frame] in the config file'
            train_set_list=[]
            validate_set_list=[]
            root_folder = data_opt['root_dir']
            for a_frame in frame_list:
                IMAGE_FORMAT_NAME = '{p_id}'+f'/sa_{a_frame}.nii.gz'
                LABEL_FORMAT_NAME = '{p_id}'+f'/label_sa_{a_frame}.nii.gz'
                IDX2CLASS_DICT = {
                    0: 'BG',
                    1: 'LV',
                    2: 'MYO',
                    3: 'RV',
                }

                train_set = Cardiac_General_Dataset(root_dir=join(root_folder,'train'),
                                                dataset_name=dataset_name,
                                                transform=tr['train'], num_classes=data_opt["num_classes"], formalized_label_dict=IDX2CLASS_DICT,
                                                idx2cls_dict=IDX2CLASS_DICT,
                                                image_format_name=IMAGE_FORMAT_NAME,
                                                label_format_name=LABEL_FORMAT_NAME,
                                                new_spacing=data_opt['new_spacing'],  
                                                use_cache=data_opt['use_cache'],
                       
                                                crop_size =  data_opt['crop_size'],
                                                normalize=True,  # disable it when data has been preprocessed to save time                                            intensity_norm_type = intensity_norm_type
                                                intensity_norm_type = intensity_norm_type
                                                ) 

                validate_set= Cardiac_General_Dataset(root_dir=join(root_folder,'validation'),
                                                dataset_name=dataset_name,
                                                transform=tr['train'], num_classes=data_opt["num_classes"], 
                                                formalized_label_dict=IDX2CLASS_DICT,
                                                idx2cls_dict=IDX2CLASS_DICT,
                                                image_format_name=IMAGE_FORMAT_NAME,
                                                label_format_name=LABEL_FORMAT_NAME,
                                                new_spacing=data_opt['new_spacing'], 
                                                crop_size =  data_opt['crop_size'],
                                                use_cache=data_opt['use_cache'],
                                                normalize=True,     # disable it when data has been preprocessed to save time                                            intensity_norm_type = intensity_norm_type
                                                intensity_norm_type = intensity_norm_type
                                                )  
    
                train_set_list.append(train_set)
                validate_set_list.append(validate_set)
            if len(frame_list) > 1:
                train_set = torch.utils.data.ConcatDataset(train_set_list)
                validate_set =torch.utils.data.ConcatDataset(validate_set_list)
            else:
                train_set = train_set_list[0]
                validate_set = validate_set_list[0]
                del train_set_list
                del validate_set_list
            
        elif 'Prostate' in dataset_name:
            train_set = ProstateDataset(transform=tr['train'],
                                        dataset_name=data_opt['dataset_name'],
                                        root_dir=data_opt["root_dir"],
                                        num_classes=data_opt["num_classes"],
                                        idx2cls_dict={0: 'BG', 1: 'FG'},
                                        use_cache=True,
                                        data_setting_name=training_opt.data_setting,
                                        split='train',
                                        cval=training_opt.cval,
                                        keep_orig_image_label_pair=True,
                                        binary_segmentation=True,
                                        new_spacing = data_opt['new_spacing'],
                                        crop_size =  data_opt['crop_size'],
                                        image_format_name=data_opt["image_format_name"],
                                        label_format_name=data_opt["label_format_name"],
                                        intensity_norm_type = intensity_norm_type
                                        )

            validate_set = ProstateDataset(transform=tr['train'],
                                        dataset_name=data_opt['dataset_name'],
                                        root_dir=data_opt["root_dir"],
                                        num_classes=data_opt["num_classes"],
                                        idx2cls_dict={0: 'BG', 1: 'FG'},
                                        use_cache=True,
                                        data_setting_name=training_opt.data_setting,
                                        split='validate',
                                        binary_segmentation=True,
                                        new_spacing = data_opt['new_spacing'],
                                        cval=training_opt.cval,
                                        crop_size =  data_opt['crop_size'],
                                        keep_orig_image_label_pair=False,
                                        image_format_name=data_opt["image_format_name"],
                                        label_format_name=data_opt["label_format_name"], 
                                        intensity_norm_type = intensity_norm_type
                                    )

        else:
            raise NotImplementedError

        datasets = [train_set, validate_set]
    else:
        datasets = []

    # ========================Define models==================================================#
    use_gpu = experiment_opt['learning']["use_gpu"] if torch.cuda.is_available() else False
    num_classes = experiment_opt['segmentation_model']["num_classes"]
    network_type = experiment_opt['segmentation_model']["network_type"]
    encoder_dropout = get_value_from_dict(experiment_opt['learning'],"encoder_dropout",None)
    decoder_dropout = get_value_from_dict(experiment_opt['learning'],"decoder_dropout",None)

    start_epoch = 0
    learning_rate = experiment_opt['learning']['lr']
    optimizer_type = get_value_from_dict(experiment_opt['learning'],"optimizer_type","Adam")
    rec_loss_type = get_value_from_dict(experiment_opt['learning'],"rec_loss_type",'l2')
    class_weights = get_value_from_dict(experiment_opt['learning'],"class_weights",None)
    separate_training = get_value_from_dict(experiment_opt['learning'],"separate_training",False)


    segmentation_solver = AdvancedTripletReconSegmentationModel(network_type=network_type,image_size=data_opt['crop_size'][0],
                                                                image_ch=1, num_classes=num_classes,
                                                                learning_rate=learning_rate,
                                                                optimizer_type=optimizer_type,
                                                                encoder_dropout = encoder_dropout,
                                                                decoder_dropout = decoder_dropout,
                                                                use_gpu=use_gpu,
                                                                n_iter=1,
                                                                checkpoint_dir=None,
                                                                debug=training_opt.debug,
                                                                rec_loss_type=rec_loss_type,
                                                                class_weights=class_weights,
                                                                separate_training=separate_training,
                                                                intensity_norm_type=intensity_norm_type
                                                                )
    if training_opt.resume_pkl_path is not None:
        start_epoch = segmentation_solver.load_snapshots(training_opt.resume_pkl_path)
        print(f'training starts at {start_epoch}')
    last_epoch = start_epoch
    
    # ========================= start training ==================================================#
    project_str = 'train_{}_{}_n_cls_{}'.format(data_opt['dataset_name'], str(
        training_opt.data_setting), str(experiment_opt['segmentation_model']["num_classes"]))
    global_dir = training_opt.save_dir
    save_dir = join(training_opt.save_dir, project_str)
    config_name = training_opt.json_config_path.replace("./config/", "")
    config_name = config_name.replace(".json", "")
    experiment_name = "{exp}/{cval}".format(exp=config_name, cval=str(training_opt.cval))
    log_dir = join(global_dir, *[project_str, experiment_name, 'log'])
    model_dir = join(global_dir, *[project_str, experiment_name, 'model'])
    check_dir(log_dir, create=True)
    check_dir(model_dir, create=True)
    ## copy  config file to saved folder
    saved_config_path   = join(global_dir, *[project_str, experiment_name, 'config.json'])
    if os.path.exists(saved_config_path):
        os.remove(saved_config_path)
    shutil.copyfile(training_opt.json_config_path,saved_config_path)


    print(f'create {model_dir} to save trained models')

    torch.cuda.empty_cache()
    if not training_opt.no_train:
        try:
            train_network(experiment_name=experiment_name,
                        dataset=datasets,
                        segmentation_solver=segmentation_solver,
                        experiment_opt=experiment_opt,
                        log=training_opt.log,
                        debug=training_opt.debug)
        except:
            print('error in training')
            if last_epoch > 0:
                save_path = segmentation_solver.save_snapshots(model_dir, epoch=last_epoch)
                print('save snapshots at epoch {} to {}'.format(str(last_epoch), save_path))
    
    if training_opt.auto_test:
        # test
        # 1. load a specific model from the specified model folder
        if training_opt.resume_pkl_path is not None or training_opt.test_model_dir_path is not None:
            assert (training_opt.resume_pkl_path is not None and training_opt.test_model_dir_path is not None) is False, "only resume_pkl_path or test_model_dir_path need to be specified"
            if training_opt.resume_pkl_path is not None:
                start_epoch = segmentation_solver.load_snapshots(training_opt.resume_pkl_path)
                print(f'loading the model from {training_opt.resume_pkl_path}, saved at epoch {start_epoch}')
                checkpoint_dir = os.path.dirname(training_opt.resume_pkl_path)

            else:
                print(f'loading model from {training_opt.test_model_dir_path}')
                segmentation_solver.get_network( training_opt.test_model_dir_path)
                checkpoint_dir = training_opt.test_model_dir_path
        else:
                # 1.1. load model from the last checkpoint
            if training_opt.use_last_epoch:
                checkpoint_dir = join(model_dir, f"{experiment_opt['learning']['n_epochs']-1}/checkpoints")
            else:
                # 1.2.  default: load model from the checkpoint with the highest vaildation accuracy
                checkpoint_dir = join(model_dir, 'best/checkpoints')
            print('load saved checkpoints from:',checkpoint_dir)
            segmentation_solver.get_network(checkpoint_dir)
        
        ## gathering test data and test model
        print ('scan test datasets')
        dataset_name = data_opt['dataset_name']
        if dataset_name == 'ACDC' or dataset_name=='UKBB':
            test_dataset_name_list = []
            test_dataset_name_list += ['ACDC', 'RandomBias', 'RandomSpike', 'RandomMotion', 'RandomGhosting']
            test_dataset_name_list += ['MSCMRSeg_C0', 'MSCMRSeg_LGE']
            test_dataset_name_list += ['MM','UKBB']
            evaluate_fn = cardiac_evaluate
            columns = ['dataset', 'method'] + ['LV (mean)', 'MYO (mean)', 'RV (mean)',
                                            'AVG'] + ['LV (std)', 'MYO (std)', 'RV (std)']
            test_root_dir_list = [""]*len(test_dataset_name_list) ## to-do 
        elif dataset_name == 'Prostate':
            test_dataset_name_list = []
            test_dataset_name_list += ['G-MedicalDecathlon']
            test_dataset_name_list += ['E-BIDMC', 'F-HK', 'A-ISBI', 'B-ISBI_1.5', 'C-I2CVB', 'D-UCL']
            evaluate_fn = prostate_evaluate
            test_root_dir_list = [data_opt["root_dir"].replace("/G-MedicalDecathlon", "")]*len(test_dataset_name_list)
            columns = ['dataset', 'method'] + ['Prostate (mean)', 'Prostate (std)']
        else:
            raise NotImplementedError

        ## summarize results    
        result_summary = []
        for test_dataset_name,test_root_dir in zip(test_dataset_name_list,test_root_dir_list):
            print (f'evaluating {test_dataset_name}')
            save_report_dir = join(checkpoint_dir, 'report')
            print('save report to:',save_report_dir)
            check_dir(save_report_dir, create=True)
            means, stds, concatenated_df = evaluate_fn(
                segmentation_model=segmentation_solver, test_dataset_name=test_dataset_name,test_root_dir=test_root_dir, test_set_ratio=1.0,
                method_name=config_name, maximum_batch_size=training_opt.test_batch_size, normalize_2D=True,new_spacing = data_opt['new_spacing'],crop_size =data_opt["crop_size"], 
                save_report_dir=save_report_dir, intensity_norm_type =intensity_norm_type )
            a_record = [test_dataset_name, config_name]
            if num_classes>2:
                ## for multi-class, add avg Dice score for ease of comparison
                means.append(np.mean(means))
            a_record.extend(means)
            a_record.extend(stds)
            result_summary.append(a_record)
        aggregated_df = pd.DataFrame(data=result_summary, columns=columns)
        print(aggregated_df)
        aggregated_df.to_csv(join(checkpoint_dir, 'report', 'dataset_summary.csv'), index=False)

