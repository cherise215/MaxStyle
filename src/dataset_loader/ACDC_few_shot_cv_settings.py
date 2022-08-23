# Created by cc215 at 28/01/20
# Enter feature description here
# Enter scenario name here
# Enter steps here
import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split


def get_ACDC_split_policy(identifier, cval):
    '''
    ACDC dataset
    the setting is the same as the one for
    "Semi-Supervised and Task-Driven Data Augmentation"
    https://arxiv.org/abs/1902.05396
    see: https://github.com/krishnabits001/task_driven_data_augmentation/blob/master/experiment_init/data_cfg_acdc.py
    :param identifier:str, the setting name for few-shot learning
    :param cval:int, id for crossvalidation
    :return: a dict of lists for training (labeled and unlabelled), testing and validate
    '''
    assert cval < 5 and cval >= 0, 'only support five fold cross validation, but got {}'.format(cval)
    # assert  identifier in['one_shot','three_shot', '40_shot'], 'invalid input, got {}'.format(identifier)
    # 20 test images
    test_list = ["007", "008", "009", "010",
                 "027", "028", "029", "030",
                 "047", "048", "049", "050",
                 "067", "068", "069", "070",
                 "087", "088", "089", "090"]

    if identifier == 'standard':
        # 70/10/20 for training and validation and test.
        training_list = ['001', '002', '003', '004', '006', '011', '012', '013', '014', '015', '016', '017', '018', '019',
                         '021', '022', '024', '025', '026', '031', '032', '033', '034', '035', '036', '038', '039', '040',
                         '041', '043', '044', '045', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                         '061', '062', '063', '064', '065', '071', '072', '073', '074', '075', '076', '077', '079', '080',
                         '081', '083', '084', '085', '086', '091', '092', '093', '094', '095', '096', '098', '099', '100']
        validate_list = ['005', '020',
                         '023', '037',
                         '042', '046',
                         '066', '078',
                         '082', '097']

        return {
            'name': str(identifier) + '_cv_' + str(cval),
            'train': training_list,
            'validate': validate_list,
            'test': test_list,
            'unlabelled': [],
            'test+unlabelled': test_list
        }

    else:
        pass
    # few-shot settings.
    validate_list = ["011", "071"]
    validation_sets = {
        0: ["062", "095", "082"],
        1: ["002", "022", "095"],
        2: ["002", "062", "095"],
        3: ["022", "062", "095"],
        4: ["022", "062", "082"]
    }
    for sid in validation_sets[cval]:
        validate_list.append(sid)

     # 25 unlabelled images
    unlabelled_list = [
        "016", "017", "018", "019", "020",
        "036", "037", "038", "039", "040",
        "056", "057", "058", "059", "060",
        "076", "077", "078", "079", "080",
        "096", "097", "098", "099", "100"]

    if 'shot' not in identifier:
        if isinstance(float(identifier), float):
            labelled_train_list = [
                "001", "002", "003", "004", "005", "006", "012", "013",
                "021", "022", "023", "024", "025", "026", "032", "033",
                "041", "042", "043", "044", "045", "046", "052", "053",
                "061", "062", "063", "064", "065", "066", "072", "073",
                "081", "082", "083", "084", "085", "086", "092", "093"]

            identifier = float(identifier)
            if 0 < identifier < 1:
                labelled_train_list, _ = train_test_split(labelled_train_list, train_size=identifier, random_state=cval)
            elif identifier >= 1:
                identifier = int(identifier)
                if 0 < identifier < len(labelled_train_list):
                    labelled_train_list, _ = train_test_split(
                        labelled_train_list, train_size=identifier, random_state=cval)
                elif identifier == len(labelled_train_list):
                    labelled_train_list = labelled_train_list
                else:
                    raise NotImplementedError
            return {
                'name': str(identifier) + '_cv_' + str(cval),
                'train': labelled_train_list,
                'validate': validate_list,
                'test': test_list,
                'unlabelled': unlabelled_list,
                'test+unlabelled': test_list + unlabelled_list

            }
        else:
            print('standard')

    else:
        # specific low-shot settings.
        if identifier == 'one_shot' or identifier == 'one_shot_upperbound':
            labelled_train_list = {
                0: ["002"],
                1: ["042"],
                2: ["022"],
                3: ["062"],
                4: ["095"]
            }[cval]
            append_validation_set = {
                0: ["042", "022", "062", "095"],
                1: ["002", "022", "062", "095"],
                2: ["002", "042", "062", "095"],
                3: ["002", "042", "022", "095"],
                4: ["002", "042", "022", "062", ],
            }[cval]
            for sid in append_validation_set:
                if sid not in validate_list:
                    validate_list.append(sid)
        elif identifier == '25_shot_upperbound':
            # 25 labelled +25 unlabelled
            train_pool = [
                "001", "002", "003", "004", "005", "006", "012", "013",
                "021", "022", "023", "024", "025", "026", "032", "033",
                "041", "042", "043", "044", "045", "046", "052", "053",
                "061", "062", "063", "064", "065", "066", "072", "073",
                "081", "082", "083", "084", "085", "086", "092", "093"]
            labelled_train_list, _ = train_test_split(train_pool, train_size=25, random_state=cval)
            labelled_train_list.extend(unlabelled_list)

        elif identifier == 'three_shot'or identifier == 'three_shot_upperbound':
            labelled_train_list = {
                0: ["002", "022", "042"],
                1: ["042", "062", "082"],
                2: ["022", "042", "082"],
                3: ["002", "042", "082"],
                4: ["002", "042", "095"]
            }[cval]
        else:
            raise NotImplementedError

        if identifier == 'three_shot_upperbound' or identifier == 'one_shot_upperbound':
            labelled_train_list = labelled_train_list + unlabelled_list

    return {
        'name': str(identifier) + '_cv_' + str(cval),
        'train': labelled_train_list,
        'validate': validate_list,
        'test': test_list,
        'unlabelled': unlabelled_list,
        'test+unlabelled': test_list + unlabelled_list
    }


def get_UKBB_split_policy(identifier, cval):
    '''
    designed for UKBB data. 500 images data, labelled from 001 to 500
    :param identifier:str, the setting name for few-shot learning
    :param cval:int, id for crossvalidation
    :return: a dict of lists for training (labeled and unlabelled), testing and validate

    '''

    # total number of images 500
    id_list = np.arange(1, 501)
    train_list = id_list[:int(500 * 0.7)]

    # 200 ublabelled images
    unlabelled_list = train_list[150:]
    validate_ind = id_list[int(500 * 0.7):int(500 * 0.8)]
    test_ind = id_list[int(500 * 0.8):]

    validate_list = ['{:03d}'.format(id) for id in validate_ind]
    test_list = ['{:03d}'.format(id) for id in test_ind]

    # total labelled pool: 150 images
    labelled_pool = train_list[:150]
    prng = RandomState(cval)
    rand_index_list = prng.permutation(len(labelled_pool))

    if identifier == '15_shot':
        labelled_train_list = ['{:03d}'.format(id) for id in rand_index_list[:15]]

    elif identifier == 'five_shot':
        labelled_train_list = ['{:03d}'.format(id) for id in rand_index_list[:5]]

    elif identifier == 'three_shot':
        labelled_train_list = ['{:03d}'.format(id) for id in rand_index_list[:3]]

    elif identifier == 'one_shot':
        labelled_train_list = ['{:03d}'.format(id) for id in rand_index_list[:1]]
    elif identifier == 'full':
        labelled_train_list = ['{:03d}'.format(id) for id in rand_index_list]
    else:
        raise NotImplementedError

    return {
        'name': identifier + '_cv_' + str(cval),
        'train': labelled_train_list,
        'validate': validate_list,
        'test': test_list,
        'unlabelled': unlabelled_list,
    }


if __name__ == '__main__':
    print(get_ACDC_split_policy('one_shot', cval=1))
    print(get_UKBB_split_policy('three_shot', cval=4))
