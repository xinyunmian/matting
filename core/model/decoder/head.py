
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadModule(nn.Module):
    def __init__(self, head_class, num_heads, **kwargs):
        super(MultiHeadModule, self).__init__()
        self.head_class = head_class
        self.num_heads = num_heads
        self._make_heads(**kwargs)

    def _make_heads(self, **kwargs):
        for n in range(self.num_heads):
            setattr(self, 'head{}'.format(n), self.head_class(**kwargs))

    def forward(self, *args, **kwargs):
        output = list()
        for n in range(self.num_heads):
            head = getattr(self, 'head{}'.format(n))
            output.append(head(*args, **kwargs))
        return output