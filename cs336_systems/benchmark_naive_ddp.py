import torch
import os
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import DDPNaive


NUM_WARMUPS = 5
NUM_STEPS = 10

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def benchmark_naive_ddp(rank, world_size, model_params, input_data, backend, num_steps):

    setup(rank, world_size, backend)

    use_gpu = torch.cuda.is_available() and backend == 'nccl'

    if use_gpu:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    model = BasicsTransformerLM(**model_params)
    model.to(device)

    ddpmodel = DDPNaive(model)

    local_batch = input_data.shape[0] // world_size
    batch_start = rank * local_batch
    batch_end = batch_start + local_batch
    input_data = input_data[batch_start:batch_end]

    if use_gpu:
        input_data = input_data.to(device)

    optimizer = torch.optim.AdamW(model.parameters())

    # warmup
    for _ in range(NUM_WARMUPS):
        optimizer.zero_grad(set_to_none=True)

        output = ddpmodel(input_data).mean()
        output.backward()

        ddpmodel.finish_gradient_synchronization()

        optimizer.step()

        if use_gpu:
            torch.cuda.synchronize()
    
    # benchmark runs
    total_sync_time = 0
    start_time = timeit.default_timer()

    for _ in range(NUM_STEPS):

        optimizer.zero_grad(set_to_none=True)

        output = ddpmodel(input_data).mean()
        output.backward()

        if use_gpu:
            torch.cuda.synchronize()

        start_sync_time = timeit.default_timer()

        ddpmodel.finish_gradient_synchronization()
        
        if use_gpu:
            torch.cuda.synchronize()
        
        end_sync_time = timeit.default_timer()
        sync_duration = end_sync_time - start_sync_time
        total_sync_time += sync_duration

        optimizer.step()

    end_time = timeit.default_timer()
    total_time = end_time - start_time
    print(f"Total training time for {num_steps} steps: {total_time}, total gradient sync time: {total_sync_time}, fraction: {total_sync_time / total_time}")

    dist.destroy_process_group()


def benchmark():
    world_size = 2
    batch_sz = 128
    backend = "nccl"
    num_steps = 10

    assert batch_sz % world_size == 0

    model_params = {
        "vocab_size": 10000,
        "context_length": 128,
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,
        "rope_theta": 10000,
    }

    input_data = torch.randint(
        low=0,
        high=10000,
        size=(batch_sz, 128),
    )

    mp.spawn(fn=benchmark_naive_ddp, args=(world_size, model_params, input_data, backend, num_steps), nprocs=world_size, join=True)


if __name__ == "__main__":
    benchmark()
