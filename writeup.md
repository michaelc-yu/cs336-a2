

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

# Problem (benchmarking_mixed_precision)

(a)
```python
initial dtype: torch.float32
dtype after first FC layer: torch.float16
dtype after relu: torch.float16
dtype after layer norm: torch.float32
dtype of logits: torch.float16
output dtype: torch.float16
loss dtype: torch.float32
fc1.weight, param dtype: torch.float32, grad dtype: torch.float32
ln.weight, param dtype: torch.float32, grad dtype: torch.float32
ln.bias, param dtype: torch.float32, grad dtype: torch.float32
fc2.weight, param dtype: torch.float32, grad dtype: torch.float32
```

* the model's parameters are in float32
* the output of the first feed-forward layer is float16
* the output of layer norm is float32
* the model's predicted logits are float16
* the loss is in float32
* the model's gradients are in float32

(b) In fp16 mixed-precision training, the autocast library still enforces that layer norm is executed in full fp32 precision. This is because layer norm could struggle with numerical stability issues if executed in fp16, specifically when calculating the variance which requires many squaring operations. The $$\epsilon$$ also might not be able to be represented with float16. 

Fp16 has more precision bits but less exponent bits, whereas bf16 has less precision bits but more exponent bits. Since bf16 has the same number of exponent bits as fp32, if using bf16 we do not need to treat layer norm differently, and we can run layer norm in bf16 format without needing to execute that step in fp32.


(c) Results in results/forward_backward_optimizer_mixed_precision.csv and results/forward_backward_with_optimizer.csv.

Mixed precision is consistently faster. For smaller models and smaller context lengths, mixed precision is only slightly faster (e.g., 0.066775 vs. 0.072913). For larger models and larger context lengths, mixed precision is much faster (e.g., 0.723469 vs. 0.388248).



