import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import seaborn as sns
from os.path import join
import time
import pickle
import logging
import torch
import scipy.misc


from src.common_utils.basic_operations import check_dir


def save_dict(mydict, file_path="./reports/summary_result.pkl"):
    f = open(file_path, "wb")
    pickle.dump(mydict, f)


def load_dict(file_path="./reports/summary_result.pkl"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_imgs(list_of_inputs, is_image=True, names=None, cmaps=None, save_dir='./result/log/adv_joint', file_name='test_{}.png'):
    '''
    plt a list of outputs from networks 
    list_of_inputs: list of input image tensors (4D)
    names for each input to show  in fig
    '''
    def get_concat_imgs_from_tensor(tensor, num_imgs=5):
        '''
        get ndarray from 4d tensor
        return concatenated 2D images (along the second axis (1)), a list of 2D imgs before concat
        '''
        if not is_image or tensor.size(1) > 1:
            tensor = torch.argmax(tensor, dim=1).unsqueeze(1)
        original_data = tensor.data.cpu().float().numpy()
        min_size = num_imgs if num_imgs <= original_data.shape[0] else original_data.shape[0]
        original_data_list = [original_data[i, 0, :, :] for i in range(min_size)]
        cat = np.concatenate(original_data_list, axis=1)
        return cat, original_data_list
    sns.set()
    fig, axes = plt.subplots(len(list_of_inputs), 1)
    print('plot start')
    for i, data in enumerate(list_of_inputs):
        cat_numpy, cat_list = get_concat_imgs_from_tensor(data, num_imgs=5)
        if cmaps is not None and len(cmaps) == len(list_of_inputs):
            cmap = cmaps[i]
            print(cmap)
        else:
            cmap = 'gray'
        if cmap == 'RdBu':
            axes[i].imshow(cat_numpy, cmap=cmap, interpolation='none', vmin=-np.max(cat_numpy), vmax=np.max(cat_numpy))
        else:
            axes[i].imshow(cat_numpy, cmap=cmap, interpolation='none')

        if names is not None and len(names) == len(list_of_inputs):
            axes[i].set_title(names[i])
        axes[i].axis('off')
    plt.tight_layout(pad=0.05,)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_mark = str(np.round(time.time(), 2))
    plt.savefig(fname=join(save_dir, file_name.format(time_mark)), dpi=500)
    print('plot finished')


def vis_results_for_different_methods(list_of_inputs, slice_id, is_image=True, names=None, cmaps=None, save_dir='./result/log/adv_joint', file_name='test.png', add_time_mark=False):
    '''
    plt a list of outputs from nrrd/nifti files 
    list_of_inputs: list of input image numpy array, [DHW], each is a volume data.
    names for each input to show  in fig
    '''
    assert list_of_inputs is not None or len(list_of_inputs) >= 1
    assert slice_id < list_of_inputs[0].shape[0], 'slice id exceeds the maxium length {}'.format(
        str(list_of_inputs[0].shape[0]))
    sns.set()
    fig, axes = plt.subplots(1, len(list_of_inputs))
    print('plot start')
    for i, data in enumerate(list_of_inputs):
        a_slice = data[slice_id]
        if cmaps is not None and len(cmaps) == len(list_of_inputs):
            cmap = cmaps[i]
            print(cmap)
        else:
            cmap = 'gray'
        if cmap is not 'gray':
            axes[i].imshow(a_slice, cmap=cmap, interpolation='none', vmin=0, vmax=3)
        else:
            axes[i].imshow(a_slice, cmap=cmap, interpolation='none')

        if names is not None and len(names) == len(list_of_inputs):
            axes[i].set_title(names[i])
        axes[i].axis('off')
    # plt.tight_layout(pad=0.01,h_pad=0)
    plt.subplots_adjust(wspace=0.01, hspace=0)
    check_dir(save_dir, create=True)
    if add_time_mark:
        time_mark = str(np.round(time.time(), 2))
        base = os.path.basename(file_name)
        prefix, extension = os.path.splitext(base)
        prefix += '_' + str(time_mark)
        file_name = prefix + extension

    try:
        plt.savefig(fname=join(save_dir, file_name), dpi=500, bbox_inches='tight', pad_inches=0)
    except e:
        logging.error(e)
    print('plot finished')
    return plt


def save_predict(img, root_dir, patient_dir, file_name):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    patient_dir = os.path.join(root_dir, patient_dir)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)
    file_path = os.path.join(patient_dir, file_name)
    sitk.WriteImage(img, file_path, True)


def save_numpy_as_nrrd(numpy_array, img_file_path):
    image = sitk.GetImageFromArray(numpy_array)
    sitk.WriteImage(image, img_file_path)


def link_image(origin_path, root_dir, patient_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    patient_dir = os.path.join(root_dir, patient_dir)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)
    image_name = origin_path.split('/')[-1]
    linked_name = image_name

    linked_path = os.path.join("\"" + patient_dir + "\"", linked_name)
    print('link path from {}  to {}'.format(origin_path, linked_path))
    os.system('ln -s {0} {1}'.format(origin_path, linked_path))


def save_results_as_png(alist, save_full_path, labels=None):
    '''
    input a list of H*W(gray)
    concat them together and save as one png
    :param alist:
    :return:
    '''

    n_length = len(alist)
    fig, ax = plt.subplots(nrows=1, ncols=n_length)  # create figure & 1 axis
    for i, img in enumerate(alist):
        if (img.max() - img.min()) > 0:
            normed_array = (((img - img.min()) / (img.max() - img.min())) * 255)
        else:
            normed_array = img
        normed_array = normed_array[:, :, None]
        normed_array = np.repeat(normed_array, axis=2, repeats=3)
        normed_array = np.uint8(normed_array)
        ax[i].imshow(normed_array)
        ax[i].axis('off')

        if labels is not None and len(labels) == n_length:
            ax[i].set_title(labels[i])
    fig.savefig(save_full_path)  # save the figure to file
    plt.close(fig)


def save_list_results_as_png(lists, save_full_path, labels=None, size=(128, 128), add_points=None, which_index=0):
    '''
    input  lists of list results H*W(gray)
    concat them together and save as one png
    :param alist:
    :return:
    '''

    n_length = len(lists)
    #print ('number of rows',n_length)
    n_cols = len(lists[0])
  #  f= plt.figure(figsize=(512, 512))
    plt.axis('tight')
    fig, ax = plt.subplots(nrows=n_length, ncols=n_cols, sharey='row', squeeze=False)
    # create figure & 1 axis
    # fig.tight_layout()
    # plt.set_aspect('equal')

    # print (ax.shape)
    for j, alist in enumerate(lists):
        for i, img in enumerate(alist):
            if (img.max() - img.min()) > 0:
                normed_array = (((img - img.min()) / (img.max() - img.min())) * 255)
            else:
                normed_array = img
            #print ('current_shape',normed_array.shape)
            normed_array = normed_array[:, :, None]
            normed_array = np.repeat(normed_array, axis=2, repeats=3)
            normed_array = np.uint8(normed_array)
            if normed_array.shape[0] > size[0] or normed_array.shape[1] > size[1]:
                # if perform downsampling, do anti-aliasing, as suggested by the official document of scikit-image
                # http://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
                anti_aliasing = True
            else:
                anti_aliasing = False
            # plt image with the same size
            normed_array = resize(normed_array, (size[0], size[1]),
                                  anti_aliasing=anti_aliasing)
            # print (normed_array.shape)
            ax[j, i].imshow(normed_array)
            if i == which_index and add_points is not None:
                from matplotlib.patches import Circle
                patches = Circle((add_points[1], add_points[0]), radius=5, color='red')
                ax[j, i].add_patch(patches)
            ax[j, i].axis('off')
            if not labels is None:
                if isinstance(labels[0], list):
                    ax[j, i].set_title(labels[j][i])
                elif labels is not None and len(labels) == n_cols:
                    ax[j, i].set_title(labels[i])
    # remove margins
    if labels is None:
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.04, hspace=0)
    else:
        fig.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.04, hspace=0)
    fig.savefig(save_full_path)  # save the figure to file
    plt.close(fig)


def save_results_with_points_as_png(alist, save_full_path, points=None, labels=None):
    '''
    input a list of H*W(gray)
    concat them together and save as one png
    : points=N*[point a, point b]
    :param alist:
    :return:
    '''

    n_length = len(alist)
    fig, ax = plt.subplots(nrows=1, ncols=n_length)  # create figure & 1 axis
    for i, img in enumerate(alist):
        if (img.max() - img.min()) > 0:
            normed_array = (((img - img.min()) / (img.max() - img.min())) * 255)
        else:
            normed_array = img
        normed_array = normed_array[:, :, None]
        normed_array = np.repeat(normed_array, axis=2, repeats=3)
        normed_array = np.uint8(normed_array)
        ax[i].imshow(normed_array)
        if not points is None:
            two_points = points[i]
            point_A = two_points[0]
            point_B = two_points[1]

            if len(point_B) == 2:
                ax[i].scatter(int(point_B[0]), int(point_B[1]), c='g', s=15)
            if len(point_A) == 2:
                ax[i].scatter(int(point_A[0]), int(point_A[1]), c='r', s=10)

        ax[i].axis('off')

        if labels is not None and len(labels) == n_length:

            ax[i].set_title(labels[i], 'center')

    fig.savefig(save_full_path)  # save the figure to file
    plt.close(fig)


def save_model(pytorch_model, save_dir, epoch_iter, model_prefix):
    '''
    save model to the disk, path '{save_dir}/{epoch_iter}/{model_prefix}.pth'
    '''
    epoch_path = join(save_dir, *[str(epoch_iter), 'checkpoints'])
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)
    torch.save(pytorch_model.state_dict(),
               join(epoch_path, model_prefix + '.pth'))


def save_testing_images_results(images, gts, predicts, save_dir, epoch_iter, max_slices=10, file_name='Seg_plots.png'):
    if epoch_iter == '':
        epoch_result_path = join(save_dir, 'predict')
    else:
        if isinstance(epoch_iter, int):
            epoch_result_path = join(save_dir, *[str(epoch_iter), 'testing_segmentation_results'])
        if isinstance(epoch_iter, str):
            epoch_result_path = join(save_dir, *[epoch_iter, 'testing_segmentation_results'])

    if not os.path.exists(epoch_result_path):
        os.makedirs(epoch_result_path)
    total_list = []
    init = True
    labels = []

    for subj_index in range(min(max_slices, gts.shape[0])):
        # for each subject
        alist = []
        temp_gt_A = gts[subj_index]
        temp_img_A = images[subj_index]
        temp_pred_A = predicts[subj_index]

        # add image and gt
        alist.append(temp_img_A)
        alist.append(temp_gt_A)
        alist.append(temp_pred_A)

        if init:
            labels.append('Input')
            labels.append('GT')
            labels.append('Predict')

        init = False

        total_list.append(alist)

    save_list_results_as_png(total_list,
                             save_full_path=join(epoch_result_path,
                                                 file_name),
                             labels=labels)


def save_model_to_file(model_name, model, epoch, optimizer, save_path):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    state = {'model_name': model_name,
             'epoch': epoch + 1,
             'model_state': state_dict,
             'optimizer_state': optimizer.state_dict()
             }
    torch.save(state, save_path)


def save_npy2image(data, file_dir, name):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filepath = os.path.join(file_dir, name + '.png')
    scipy.misc.imsave(filepath, data)


def save_medical_image(itkImage, output_path, refImage=None):
    if refImage is not None:
        itkImage.CopyInformation(refImage)
    sitk.WriteImage(itkImage, output_path)


def save_nrrd_to_disk(save_folder, file_name, image, pred, gt):
    '''
    save image, gt, pred of a patient as nrrd files under a specified folder
    :param save_folder: string
           folder path
    :param file_name: string
        unique id as file prefix
    :param image: ndarray
        N*H*W
    :param gt: ndarray
        N*H*W
    :param pred: ndarray
        N*H*W
    '''

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_img_path = os.path.join(save_folder, file_name + "_image.nrrd")
    save_gt_path = os.path.join(save_folder, file_name + "_label.nrrd")
    save_pred_path = os.path.join(save_folder, file_name + "_pred.nrrd")

    save_numpy_as_nrrd(image, save_img_path)
    save_numpy_as_nrrd(pred, save_pred_path)
    save_numpy_as_nrrd(gt, save_gt_path)
