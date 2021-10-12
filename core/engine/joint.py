
import os
import torch
import time
import numpy as np

from core.engine.base import Engine
from core.model.network import *
from core.loss.joint import *
from core.metric.miou import *
from core.dataset import *



class EngineJHMP(Engine):
    def __init__(self, yaml_path):
        super(EngineJHMP, self).__init__()
        self.config = self.load_configuration(yaml_path)
        self.config_object()

    def config_object(self):
        from torch.utils.data import DataLoader
        self.dataset = DataSetDict[self.config['dataset']](config=self.config)
        self.worker = DataLoader(dataset=self.dataset, batch_size=self.config['batch_size'], drop_last=True, shuffle=True,
                                 num_workers=self.config['num_workers'], pin_memory=self.config['pin_memory'])
        self.network = JointMattingParsing(backbone=self.config['backbone'], num_classes=self.config['num_classes'])
        self.loss_function = JointMattingParsingLoss(phase=self.config['phase'])
        self.metric_segment = MeanInterSectionOverUnion(num_classes=self.config['num_classes'])

        self.optimizer = torch.optim.SGD(
            [{'params': self.network.parameters()}, {'params': self.loss_function.parameters()}],
            momentum=self.config['momentum'], weight_decay=self.config['weight_decay'], lr=self.config['beg_lr'])
        self.learn_rate = self.config['beg_lr']
        self.device = None  # torch.device(self.config['device'])
        # config other options
        self.config_device(self.config['device'])
        self.config_data_parallel()

    def move_to_gpu(self, batch:{str:torch.tensor}):
        import random
        base_size = 128
        prefix = random.randint(0, 128)
        add = random.randint(0, (128-prefix)//16)
        beg = prefix
        end = beg + 128 + add*16
        assert beg <= 128 and end <= 256
        for key in batch:
            batch[key] = batch[key][:, :, :, beg:end].cuda(self.device)
            # batch[key] = batch[key].cuda(self.device)

    def train_one_epoch(self, **kwargs):
        self.module.train()

        train_loss_list = list()
        train_miou_list = list()
        train_loss_seg_list = list()
        train_loss_mat_list = list()

        start = time.time()
        for n, batch in enumerate(self.worker):
            self.move_to_gpu(batch=batch)

            output = self.module(batch['image'])
            target = dict(segment=batch['segment'], matting=batch['matting'])
            loss, loss_segment, loss_matting = self.loss_function(output=output, target=target)
            miou = self.metric_segment(output=output, target=target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss_list.append(loss.item())
            train_miou_list.append(miou.item())
            train_loss_seg_list.append(loss_segment.item())
            train_loss_mat_list.append(loss_matting.item())
            # break
        elapsed = time.time() - start

        info_dict = {
            'loss': float(np.array(train_loss_list, dtype=np.float32).mean()),
            'loss_segment': float(np.array(train_loss_seg_list, dtype=np.float32).mean()),
            'loss_matting': float(np.array(train_loss_mat_list, dtype=np.float32).mean()),
            'miou': float(np.array(train_miou_list, dtype=np.float32).mean())
        }

        # output information
        format = self.information_line.format(
            epoch=kwargs['epoch'],
            num_epoch=self.config['num_epochs'],
            time=elapsed,
            loss=info_dict['loss'],
            loss_segment=info_dict['loss_segment'],
            loss_matting=info_dict['loss_matting'],
            miou=info_dict['miou'],
            learn_rate=self.learn_rate,
        )
        print(format)
        self.log_file.write(format + '\n')
        self.log_file.flush()

        if kwargs['epoch'] % self.config['save_epoch'] == 0:
            self.save_checkpoint(prefix=self.config['prefix_name'], suffix=kwargs['epoch'])

    def train(self, **kwargs):
        if not self.config['backbone_pretrain'] is None:
            self.network.load_backbone_state_dict(self.config['backbone_pretrain'])
            print('============> load backbone pre-train: {}'.format(self.config['backbone_pretrain']))

        if not self.config['pretrain'] is None:
            state_dict = torch.load(self.config['pretrain'])
            self.network.load_state_dict(state_dict)
            print('============> load whole pre-train: {}'.format(self.config['pretrain']))

        if torch.cuda.is_available():
            print('============> using CUDA at GPU({})'.format(self.device))
            self.module.cuda(self.device)
            self.metric_segment.cuda(self.device)
            self.loss_function.cuda(self.device)
            if self.config['benchmark'] is True:
                print('============> enable cudnn benchmark')
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

        if os.path.exists(self.config['save_path']) == False:
            os.mkdir(self.config['save_path'])

        self.log_file = open('{}/train.txt'.format(self.config['save_path']), 'w')
        self.dump_configuration(self.log_file, self.config)

        self.checkpoint_path = os.path.join(self.config['save_path'], '{prefix}-{suffix}.pth')
        self.information_line = '[{epoch:4d}/{num_epoch:4d}, {time:.1f}s] ==> ' \
                                '{loss:.6f}, {loss_segment:.6f}, {loss_matting:.6f}, {miou:.4f}, {learn_rate:.6f}'

        for epoch in range(1, self.config['num_epochs'] + 1):
            self.train_one_epoch(epoch=epoch)
            self.adjust_learning_rate(self.optimizer, epoch, self.config)

        self.save_checkpoint(prefix=self.config['prefix_name'], suffix='final')

    def save_checkpoint(self, prefix, suffix):
        path = self.checkpoint_path.format(prefix=prefix, suffix=suffix)
        torch.save(self.network.state_dict(), path)

    def valid(self, **kwargs):
        if 'checkpoint' in kwargs:
            checkpoint_path = kwargs['checkpoint']
        else:
            assert os.path.exists(self.config['save_path'])
            checkpoint_path = '{}/{}-{}.pth'.format(
                self.config['save_path'], self.config['prefix_name'], 'final')

        assert os.path.isfile(checkpoint_path)
        state_dict = torch.load(checkpoint_path)
        self.network.load_state_dict(state_dict)
        print('============> load checkpoint: {}'.format(checkpoint_path))

        if torch.cuda.is_available():
            print('============> using CUDA at GPU({})'.format(self.device))
            self.module.to(self.device)
            self.metric_segment.to(self.device)
            self.loss_function.to(self.device)
            if self.config['benchmark'] is True:
                print('============> enable cudnn benchmark')
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

        log_file = open('{}/valid.txt'.format(self.config['save_path']), 'w')

        self.module.eval()
        detach = lambda x: x.cpu().detach().numpy()

        sad_list = list()
        mse_list = list()
        ce_list = list()
        ge_list = list()
        iou_list = list()
        for n, batch in enumerate(self.worker):
            image = batch['image'].cuda(self.device)
            segment = batch['segment'].cuda(self.device)
            matting = batch['matting'].cuda(self.device)
            output = self.module(image)
            target = dict(segment=segment, matting=matting)
            iou_list.append(self.metric_segment(output=output, target=target).item())
            # print(segment.shape, matting.shape, output['matting'].shape)
            # np_mat_gt = detach(matting)[0, 0, :, :] * 255.
            # np_mat_pr = detach(output['matting'])[0, 0, :, :] * 255.
            # np_tri = np.ones_like(np_mat_gt) * 128
            # sad_list.append(comput_sad_loss(pred=np_mat_pr, target=np_mat_gt, trimap=np_tri))
            # mse_list.append(compute_mse_loss(pred=np_mat_pr, target=np_mat_gt, trimap=np_tri))
            # ce_list.append(compute_connectivity_error(pred=np_mat_pr, target=np_mat_gt, trimap=np_tri, step=0.1))
            # ge_list.append(compute_gradient_loss(pred=np_mat_pr, target=np_mat_gt, trimap=np_tri))
            log_file.write('{:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(
                sad_list[-1], mse_list[-1], ce_list[-1], ge_list[-1], iou_list[-1]))

        size = 512 * 512
        avg_sad = np.array(sad_list).mean() / size
        avg_mse = np.array(mse_list).mean() / size
        avg_ce = np.array(ce_list).mean() / size
        avg_ge = np.array(ge_list).mean() / size
        avg_iou = np.array(iou_list).mean()
        line = '{:.8f} {:.8f} {:.8f} {:.8f} {:.8f}'.format(
            avg_sad, avg_mse, avg_ce, avg_ge, avg_iou)
        log_file.write(line + '\n'), print(line)


