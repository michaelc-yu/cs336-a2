import torch
from torch import nn
import torch.distributed as dist


class DDPNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        for param in module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

