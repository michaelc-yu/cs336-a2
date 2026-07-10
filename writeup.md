

# Problem (benchmarking_script)

(a) Code in benchmark.py. Results in results/

(b) Forward + backward pass takes about 3x as long as forward only. The times are on the scale of 1/100 of a second to almost a second. The variability across the measurements is pretty small, stdev is often under 1% or only slightly above.

(c) With no warmup, the times are only slightly higher but the stdev is much larger. Potentially the first 1 or 2 steps there is a lot of kernel / caching work done and is much slower, therefore increasing the stdev. All steps after that benefit from the system having already warmed up.
With one warmup step, the variability is much lower but still slightly higher than 5 warmup steps. Also the mean times are very similar to 5 warmup steps. Could be that the first step handled much of the warmup but not all of it.

# Problem (nsys_profile)

Seems like LambdaLabs gpu instances currently don't support nsight profiling. So will skip for now.

# Problem (mixed_precision_accumulation)

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)
```
Prints tensor(10.0001)

```python
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)
```
Prints tensor(9.9531, dtype=torch.float16)

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)
```
Prints tensor(10.0021)

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
```
Prints tensor(10.0021)

fp32 for both the accumulator and incremental values performs best. The precision of the accumulator is more important than the precisions of each increment.


