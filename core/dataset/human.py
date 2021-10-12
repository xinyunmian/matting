
import os
import numpy as np
import cv2
import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from core.dataset.trimap import TrimapGeneration, TrimapGenerator


class HumanMattingAndSegmentation(Dataset):
    def __init__(self, config):
        self._parse_config(config)
        self._build_augment()
        self.tri_gen = TrimapGenerator()

    def _parse_config(self, config):
        self.crop_height = config['crop_width']
        self.crop_width = config['crop_width']
        self.path_img, self.path_seg, self.path_mat = list(), list(), list()
        if 'dataset_yaml' in config:
            dataset_dict = yaml.load(open(config['dataset_yaml'], 'r'))
            for name in dataset_dict:
                img_list, seg_list, mat_list = self._parse_dataset(dataset_dict[name], name)
                self.path_img.extend(img_list)
                self.path_seg.extend(seg_list)
                self.path_mat.extend(mat_list)
        else:
            pass
        print('============> total load {} images...'.format(len(self.path_img)))

    def _parse_dataset(self, config, name):
        list_file = '{}/{}.{}'.format(config['root_dataset'], name, config['train_list'])
        name_list = self._load_list(list_file)
        img_list, seg_list, mat_list = list(), list(), list()
        path_img = '{}/{}'.format(config['root_dataset'], config['images_path'])
        path_seg = '{}/{}'.format(config['root_dataset'], config['segment_path'])
        path_mat = '{}/{}'.format(config['root_dataset'], config['matting_path'])
        for name in name_list:
            img_list.append('{}/{}.png'.format(path_img, name))
            seg_list.append('{}/{}.png'.format(path_seg, name))
            mat_list.append('{}/{}.png'.format(path_mat, name))
        return img_list, seg_list, mat_list

    def _load_list(self, file_path):
        with open(file_path, 'r') as file:
            file_list = list()
            for line in file:
                file_list.append(line.rstrip('\n'))
            print('============> load {} images from {}'.format(len(file_list), file_path))
        return file_list

    def _build_augment(self):
        self.augment = iaa.Sequential([
            iaa.Fliplr(0.5),
            # iaa.PadToSquare(pad_cval=0),
            iaa.SomeOf(1, [
                iaa.Affine(rotate=(-15, 15)),
                iaa.Affine(shear=(-5, 5)),
            ]),
            # iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.75, 1.5)),
            iaa.Resize((0.25, 2)),
            iaa.Resize({'height': (0.8, 1.2), 'width': (0.8, 1.2)}),
            iaa.PadToSquare(pad_cval=0),
            iaa.Resize({'height': self.crop_height+32,'width': self.crop_width+32}),
            iaa.CenterCropToFixedSize(self.crop_width, self.crop_height),
        ])

    def __len__(self):
        return len(self.path_img)

    def __getitem__(self, item):
        img = cv2.imdecode(np.fromfile(self.path_img[item], dtype=np.uint8), cv2.IMREAD_COLOR)
        mat = cv2.imdecode(np.fromfile(self.path_mat[item], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert img.shape[:2] == mat.shape[:2]
        # do augment
        img, mat = self._do_augment(img, mat)
        # generate trimap
        seg = self.tri_gen(mat)
        # convert to Tensor
        img_th = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.
        seg_th = torch.from_numpy(seg).view(1, self.crop_height, self.crop_width).long()
        mat_th = torch.from_numpy(mat).view(1, self.crop_height, self.crop_width).float() / 255.
        return {'image': img_th, 'segment': seg_th, 'matting': mat_th}

    def _do_augment(self, img, mat):
        H, W, _ = img.shape
        img = np.reshape(img, (H, W, 3))
        mat = np.reshape(mat, (H, W, 1))
        concat = np.concatenate([img, mat], axis=2)
        aug_concat = self.augment(image=concat)
        return aug_concat[:, :, :3], aug_concat[:, :, 3]



if __name__ == '__main__':
    import yaml
    config = yaml.load(open('config/local.yaml', 'r'))
    dataset = HumanMattingAndSegmentation(config=config)
    worker = DataLoader(dataset=dataset, batch_size=2, drop_last=False, shuffle=True, num_workers=1, pin_memory=False)
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
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR) * 127
        mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
        # ext = cv2.cvtColor(np.reshape(ext, (512, 512)), cv2.COLOR_GRAY2BGR)
        vis = np.concatenate([img, seg, mat], axis=1)
        cv2.imshow('show', vis)
        cv2.waitKey(0)
    # collect_all_dataset(config)