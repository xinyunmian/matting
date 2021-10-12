
import os
import math
import json
import yaml
import pprint


class Engine:
    def __init__(self):
        self.config = None
        self.module = None
        self.network = None
        self.dataset = None
        self.worker = None

    def get_host_ip(self):
        import socket
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return ip

    def load_configuration(self, yaml_path):
        assert os.path.exists(yaml_path)
        print('load config file: {}'.format(yaml_path))
        config = yaml.load(open(yaml_path, 'r'))
        pprint.pprint(config, compact=True)
        return config

    def config_device(self, device_id):
        assert hasattr(self, 'device_id') is False
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        self.device_id = device_id

    def config_data_parallel(self):
        import torch
        assert hasattr(self, 'network') is True
        assert hasattr(self, 'device_id') is True
        if len(self.device_id.split(',')) > 1 and torch.cuda.device_count() > 1:
            print('============> data parallel with GPU: {}'.format(self.device_id))
            self.module = torch.nn.DataParallel(self.network)
        else:
            self.module = self.network

    def config_distribute_with_tcp(self):
        import torch.distributed as dist
        import torch.utils.data.distributed
        host_ip = self.get_host_ip()
        print('============> distribute training: \n'
              '\tmaster: {}'
              '\thost: {}(rank {}) with GPU {}'.format(
            self.config['init_method'], host_ip, self.config['rank'], self.device_id))
        dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(host_ip, self.config['port']),
                                rank=self.config['rank'], world_size=self.config['word_size'])
        assert hasattr(self, 'dataset') is True
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        self.worker = self.build_worker(self.dataset, self.sampler)
        assert hasattr(self, 'network') is True
        assert hasattr(self, 'device_id') is True
        self.module = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=self.device_id)

    def config_distribute_with_env(self):
        pass

    def adjust_learning_rate(self, opt, epoch, cfg):
        if epoch <= cfg['adj_epochs']:
            multi = math.pow(1 - epoch / cfg['adj_epochs'], 2)
            self.learn_rate = (cfg['beg_lr'] - cfg['end_lr']) * multi + cfg['end_lr']
        else:
            self.learn_rate = cfg['end_lr']
        # usually: 0 is network, 1 is weight for different loss
        for param in opt.param_groups:
            param['lr'] = self.learn_rate

    def dump_configuration(self, file, config):
        file.write(json.dumps(config, indent=4) + '\n')

    def build_worker(self, dataset, sampler=None):
        from torch.utils.data import DataLoader
        assert 'batch_size' in self.config
        assert 'num_workers' in self.config
        assert 'pin_memory' in self.config
        worker = DataLoader(dataset=dataset, batch_size=self.config['batch_size'], drop_last=True, shuffle=True,
                            num_workers=self.config['num_workers'], pin_memory=self.config['pin_memory'], sampler=sampler)
        return worker

    def train(self):
        ...

    def eval(self):
        ...

    def run(self, **kwargs):
        assert 'mode' in kwargs
        if kwargs['mode'] is 'train':
            print('mode is train')
            self.train()
        else:
            self.eval()