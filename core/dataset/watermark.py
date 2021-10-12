
import os
import numpy as np
import cv2
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from core.dataset.trimap import TrimapGeneration


class WaterMark(Dataset):
    def __init__(self, config):
        path = '{}/{}'.format(config['root_dataset'], config['train_list'])
        # self.mode = config['mode']
        self.file_list = self._load_file_list(path)
        self.path_img = '{}/{}'.format(config['root_dataset'], config['images_path'])
        self.path_seg = '{}/{}'.format(config['root_dataset'], config['segment_path'])
        self.path_mat = '{}/{}'.format(config['root_dataset'], config['matting_path'])
        self.path_src = '{}/{}'.format(config['root_dataset'], config['source_path'])

        self.crop_height = config['crop_height']
        self.crop_width = config['crop_width']
        self.tri_gen = TrimapGeneration()
        self.augment = iaa.Sequential([
            iaa.Sometimes(
                0.9,
                iaa.OneOf([
                    iaa.ChannelShuffle(p=0.8),
                    iaa.OneOf([
                        iaa.GaussianBlur(sigma=1.5),
                        iaa.MedianBlur(k=7),
                        iaa.AverageBlur(k=7),
                    ]),
                    iaa.OneOf([
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5),
                        iaa.Dropout(p=(0.0, 0.08)),
                    ]),
                ]),
            ),
        ])

    def _load_file_list(self, file_path):
        with open(file_path, 'r') as file:
            file_list = list()
            for line in file:
                file_list.append(line.rstrip('\n'))
            print('============> load {} images from {}'.format(len(file_list), file_path))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name = self.file_list[item]
        img = cv2.imdecode(np.fromfile(os.path.join(self.path_img, name + '.png'), dtype=np.uint8), -1)
        mat = cv2.imdecode(np.fromfile(os.path.join(self.path_mat, name + '.png'), dtype=np.uint8), -1)
        img, mat = self._format(img, mat)
        seg = np.where(mat > 127., np.ones_like(mat), np.zeros_like(mat))
        # format
        H, W, _ = img.shape
        img = np.reshape(img, (H, W, 3))
        seg = np.reshape(seg, (H, W, 1))
        mat = np.reshape(mat, (H, W, 1))
        img = self.augment(image=img)
        # convert to Tensor
        img_th = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.
        seg_th = torch.from_numpy(seg).view(1, self.crop_height, self.crop_width).long()
        mat_th = torch.from_numpy(mat).view(1, self.crop_height, self.crop_width).float() / 255.
        # seg_th = self.tri_gen(mat_th)
        return {'image':img_th, 'segment':seg_th, 'matting':mat_th}

    def _format(self, img, mat):
        img = cv2.resize(img, (self.crop_width, self.crop_height), interpolation=cv2.INTER_LINEAR)
        mat = cv2.resize(mat, (self.crop_width, self.crop_height), interpolation=cv2.INTER_LINEAR)
        return img, mat




class WaterMarkWorker(DataLoader):
    def __init__(self, config, **kwargs):
        dataset = WaterMark(config=config)
        super(DataLoader, self).__init__(dataset=dataset, **kwargs)




if __name__ == '__main__':
    import yaml
    config = yaml.load(open('config/local.yaml', 'r'))
    dataset = WaterMark(config=config)
    worker = DataLoader(dataset=dataset, batch_size=1, drop_last=False, shuffle=True, num_workers=1, pin_memory=False)
    # metric = get_metric_object(3)
    detach = lambda t: t.cpu().detach().numpy()
    for n, batch in enumerate(worker):
        image = batch['image']
        segment = batch['segment']
        matting = batch['matting']
        segment_fgd = (segment == 2).long()
        segment_bgd = (segment == 0).long()
        # print(metric(target={'segment':segment}, output={'segment':segment}))
        img = np.transpose(detach(image*255)[0,0:3,:,:], axes=(1, 2, 0)).astype(np.uint8)
        # ext = np.transpose(detach(image*255)[0,3:4,:,:], axes=(1, 2, 0)).astype(np.uint8)
        seg = detach(segment)[0,0,:,:].astype(np.uint8)
        mat = detach(matting*255)[0,0,:,:].astype(np.uint8)
        print(img.shape, seg.shape, mat.shape)
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR) * 255
        mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
        # ext = cv2.cvtColor(np.reshape(ext, (512, 512)), cv2.COLOR_GRAY2BGR)
        vis = np.concatenate([img, seg, mat], axis=1)
        cv2.imshow('show', vis)
        cv2.waitKey(0)