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

class DDPBatch(DDPNaive):

    def finish_gradient_synchronization(self):
        gradients = [param.grad for param in self.module.parameters() if param.grad is not None]

        flat_grads = torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= dist.get_world_size()

        synced_grads = torch._utils._unflatten_dense_tensors(flat_grads, gradients)

        for original_grad, synced_grad in zip(gradients, synced_grads):
            original_grad.copy_(synced_grad)

