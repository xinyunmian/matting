
import torch
import torch.nn as nn



class MultiTaskProxy(nn.Module):
    def __init__(self, num_task):
        super(MultiTaskProxy, self).__init__()
        self.num_task = num_task
        self.log_vars = nn.Parameter(torch.zeros(num_task), requires_grad=True)

    def forward(self, loss_list:list):
        assert len(loss_list) == self.num_task
        loss = 0
        var = torch.exp(-self.log_vars)
        for n, one in enumerate(loss_list):
            loss += one * var[n] + self.log_vars[n]
        return loss