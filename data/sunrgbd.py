"""
Created on Mar, 2018

@author: Siyuan Huang

Preprocess the SUNRGBD dataset
"""
import torch
import json
import pickle
import config
import os.path as op
import numpy as np
from PIL import Image
import collections
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms

PATH = config.Config('sunrgbd')
HEIGHT_PATCH = 256
WIDTH_PATCH = 256

# transforms.ToTensor


class SUNRGBDDataset(Dataset):
    def __init__(self, list_file, random_flip=False, random_shift=False, random_shift_times=5):
        """
        Args:
            list_file: data list
        """
        with open(list_file, 'r') as f:
            self.data_frame = json.load(f)
        f.close()
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.random_shift_times = random_shift_times

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        rand_num = np.random.rand()
        file_path = self.data_frame[index]
        if self.random_flip is True and rand_num > 0.5:
            is_flip = True
        else:
            is_flip = False
        if is_flip:
            file_path = file_path[:-7] + '_flip.pickle'
        if self.random_shift:
            shift_rand = np.random.randint(self.random_shift_times+1)
            if shift_rand > 0:
                file_path = file_path[:-7] + '_shift_' + str(shift_rand) + '.pickle'
        with open(file_path, 'r') as f:
            sequence = pickle.load(f)
        f.close()
        image = Image.open(sequence['rgb_path']).convert('RGB')
        camera = sequence['camera']
        boxes = sequence['boxes']
        layout = sequence['layout']
        patch = list()
        data_transforms_nocrop = transforms.Compose([
                transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        data_transforms_crop = transforms.Compose([
                transforms.Resize((280, 280)),
                transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        for bdb in boxes['bdb_pos']:
            img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            img = data_transforms_crop(img)
            patch.append(img)
        boxes['patch'] = torch.stack(patch)
        image = data_transforms_nocrop(image)
        return {'image': image, 'boxes_batch': boxes, 'camera': camera, 'layout': layout, 'sequence_id': sequence['sequence_id']}


def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem


default_collate = torch.utils.data.dataloader.default_collate


def collate_fn(batch):
    """
    SUNRGBD data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batch: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:
        if key == 'boxes':
            collated_batch[key] = [recursive_convert_to_torch(elem[key]) for elem in batch]
        elif key == 'boxes_batch':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                tensor_batch = torch.cat(list_of_tensor)
                collated_batch[key][subkey] = tensor_batch
        else:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
    return collated_batch


def sunrgbd_train_loader(opt):
    return DataLoader(dataset=SUNRGBDDataset(op.join(opt.metadataPath, opt.dataset, 'train.json'), random_flip=True, random_shift=False),
                      num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, collate_fn=collate_fn)


def sunrgbd_test_loader(opt):
    return DataLoader(dataset=SUNRGBDDataset(op.join(opt.metadataPath, opt.dataset, 'test.json'), random_flip=False, random_shift=False),
                      num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, collate_fn=collate_fn)


