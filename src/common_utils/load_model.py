import os
from os.path import join

import torch


def resume_model_from_file(file_path):
    start_epoch = 1
    optimizer_state = None
    state_dict = None
    checkpoint = None
    assert os.path.isfile(file_path)
    if '.pkl' in file_path:
        print("Loading models and optimizer from checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        for k, v in checkpoint.items():
            if k == 'model_state':
                state_dict = checkpoint['model_state']
            if k == 'optimizer_state':
                optimizer_state = checkpoint['optimizer_state']
            if k == 'epoch':
                start_epoch = int(checkpoint['epoch'])
        print("Loaded checkpoint '{}' (epoch {})"
              .format(file_path, checkpoint['epoch']))
    elif '.pth' in file_path:
        print("Loading models and optimizer from checkpoint '{}'".format(file_path))
        state_dict = torch.load(file_path)
        start_epoch = int(file_path.split('.')[0].split('_')[-1])  # restore training procedure.
    else:
        raise NotImplementedError

    return {'start_epoch': start_epoch,
            'optimizer_state': optimizer_state,
            'state_dict': state_dict,
            'checkpoint': checkpoint
            }


def restoreOmega(path, model, optimizer=None):
    checkpoint = resume_model_from_file(file_path=path)
    state_dict = checkpoint['state_dict']
    start_epoch = checkpoint['start_epoch']
    model.load_state_dict(state_dict, strict=False)
    optimizer_state = checkpoint['optimizer_state']
    if not (optimizer_state is None) and (not optimizer is None):
        try:
            optimizer.load_state_dict(optimizer_state)
        except:
            pass
    return model, optimizer, start_epoch
