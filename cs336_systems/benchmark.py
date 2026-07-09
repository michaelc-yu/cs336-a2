
import argparse
from cs336_basics.model import BasicsTransformerLM
import torch
import timeit
import statistics


def benchmark_model(
    model_params,
    input_data,
    num_warmups,
    num_trials,
    forward_only,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BasicsTransformerLM(**model_params)
    model.to(device)
    input_data = input_data.to(device)

    def execute_step():
        output = model(input_data).mean()
        if not forward_only:
            output.backward()


    for _ in range(num_warmups):
        execute_step()
    
    times = []
    for _ in range(num_trials):
        start_time = timeit.default_timer()
        
        execute_step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    # return the average and standard deviation of times
    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_warmups", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--forward_only", action="store_true")

    args = parser.parse_args()
    print(args)

    model_params = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }

    input_data = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
    )

    res = benchmark_model(
        model_params,
        input_data,
        args.num_warmups,
        args.num_trials,
        args.forward_only,
    )
    print(f"Mean time per step: {res['mean']:.6f} s")
    print(f"Std: {res['std']:.6f} s")


if __name__ == "__main__":
    main()

