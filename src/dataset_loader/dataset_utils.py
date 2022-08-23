import SimpleITK as sitk
import numpy as np


def sample_batch_from_dataloader(dataiter, dataloader):
    # for i, batch in cycle(enumerate(data_loader)):
    try:
        batch = next(dataiter)
    except StopIteration:
        dataiter = dataloader.__iter__()
        batch = next(dataiter)
    return batch


def normalize_minmax_data(image_data, min_percentile=2, max_percentile=98):
    """
    # 3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
    Here, the minimum and maximum values are used as 2nd and 98th percentiles respectively from the 3D MRI scan.
    We expect the outliers to be away from the range of [0,1].
    input params :
        image_data : 3D MRI scan to be normalized using min-max normalization
    returns:
        final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
    """
    min_val_2p = np.percentile(image_data, min_percentile)
    max_val_98p = np.percentile(image_data, max_percentile)
    final_image_data = np.zeros(
        (image_data.shape[0], image_data.shape[1], image_data.shape[2]), dtype=np.float32)
    # min-max norm on total 3D volume
    image_data[image_data < min_val_2p] = min_val_2p
    image_data[image_data > max_val_98p] = max_val_98p

    final_image_data = (image_data - min_val_2p) / (1e-10 + max_val_98p - min_val_2p)

    return final_image_data


def resample_by_spacing(im, new_spacing, interpolator=sitk.sitkLinear, keep_z_spacing=False):
    '''
    resample by image spacing
    :param im: sitk image
    :param new_spacing: new image spacing x,y,z
    :param interpolator: sitk.sitkLinear, sitk.NearestNeighbor
    :return:
    '''

    scaling = np.array(new_spacing) / (1.0 * (np.array(im.GetSpacing())))
    new_size = np.round((np.array(im.GetSize()) / scaling)).astype("int").tolist()
    origin_z = im.GetSize()[2]
    old_spacing  = im.GetSpacing()
    if keep_z_spacing:
        new_size[2] = origin_z
        scaling[2] = 1
        new_spacing[2] = old_spacing[2]
    if  new_size[2] == origin_z:
        # print('shape along z axis does not change')
        pass
    if abs(np.sum(np.array(scaling)) - len(scaling)) < 1e-4:
        return im
    else:
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(im.GetDirection())
        resample.SetOutputOrigin(im.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(im.GetPixelIDValue())
        resample.SetInterpolator(interpolator)
        # print ('resampling image with spacing: ', new_spacing)
        return resample.Execute(im)


def resample_by_ref(im, refim, interpolator=sitk.sitkLinear):
    transform = sitk.Transform()
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(im, refim, transform, interpolator)


def get_all_image_array_from_datastet(dataset, crop_size=[192, 192]):
    img_list = None
    for id in range(dataset.patient_number):
        data = dataset.get_patient_data_for_testing(id, crop_size=crop_size)
        img_array_flattened = data['image'].numpy()
        img_array_flattened = img_array_flattened.reshape(img_array_flattened.shape[0], -1).squeeze()
        if img_list is None:
            img_list = img_array_flattened
        else:
            img_list = np.concatenate((img_list, img_array_flattened), axis=0)

    image_arrays = np.array(img_list)
    image_arrays = image_arrays.reshape(1, -1)
    return image_arrays


def get_mean_image(dataset):
    image_arrays = get_all_image_array_from_datastet(dataset)
    return np.mean(image_arrays, axis=0)
