

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


# Problem (memory_profiling)

(a)
Full training step with context length 128
![memory_profile_train_step](./results/memory_profiling/forward_backward_optimizer_ctx_128.png)

Forward only with context length 128
![memory_profile_forward](./results/memory_profiling/forward_ctx_128.png)

The forward only image has consistent sharp spikes then decreases. The sharp spikes are likely the forward passes. And since there is no backward pass, the activations are freed. For the full training step, the activations are not fully freed after each forward pass. Furthermore, there are some thin sharp spikes which could correspond to the optimizer steps.


(b) Peak memory usage of forward step by context length:
* 128: 6.6 GB
* 256: 10.3 GB
* 512: 19.8 GB

Full training step:
* 128: 18 GB

(c) Mixed-precision peak memory usage of forward step by context length:
* 128: 7.2 GB
* 256: 9.6 GB
* 512: 15.8 GB

Full training step: 
* 128: 17.9 GB

Using mixed-precision can reduce memory usage at larger context lengths, but sometimes even increases memory usage at smaller context lengths. At larger context lengths, activation memory dominates so the memory savings from mixed-precision becomes significant. At smaller context lengths, the additional overhead from maintaining mixed-precision states and conversions can outweigh activation savings.

(d) For the model with context length 128, batch size of 4, and d_model = 1600:

* activation size = (B x T x d_model x bytes per element) / (1024^2)
* activation size = (4 x 128 x 1600 x 4) / (1024^2)
* activation size = 3.125 MiB

(e) When reducing detail to only 10%, the largest allocation is of size 80 MiB. 
Looking at the stacktrace, it looks like the allocation comes from tensor division operations (div_Tensor, PyNumber_TrueDivide). These could be temporary tensors created during layer norm or attention scaling operations.


# Problem (pytorch_attention)

| d_model | seq_len | fwd_time (s) | bwd_time (s) | mem after fwd (MB) | status |
|---------|---------|--------------|--------------|---------------------|--------|
| 16      | 256     | 0.0325       | 0.0608       | 21.21               | OK     |
| 16      | 1024    | 0.0725       | 0.1672       | 84.84               | OK     |
| 16      | 4096    | 0.7568       | 1.8534       | 1070.63             | OK     |
| 16      | 8192    | 2.8289       | 6.8483       | 4205.00             | OK     |
| 16      | 16384   |              |              |                     | OOM    |
| 32      | 256     | 0.0309       | 0.0621       | 22.09               | OK     |
| 32      | 1024    | 0.0652       | 0.1617       | 88.34               | OK     |
| 32      | 4096    | 0.7833       | 1.8849       | 1084.63             | OK     |
| 32      | 8192    | 2.9185       | 6.9421       | 4233.00             | OK     |
| 32      | 16384   |              |              |                     | OOM    |
| 64      | 256     | 0.0309       | 0.0621       | 23.84               | OK     |
| 64      | 1024    | 0.0682       | 0.1657       | 95.34               | OK     |
| 64      | 4096    | 0.8284       | 1.9336       | 1112.63             | OK     |
| 64      | 8192    | 3.1000       | 7.1270       | 4289.00             | OK     |
| 64      | 16384   |              |              |                     | OOM    |
| 128     | 256     | 0.0309       | 0.0582       | 27.34               | OK     |
| 128     | 1024    | 0.0741       | 0.1726       | 109.34              | OK     |
| 128     | 4096    | 0.9141       | 2.0257       | 1168.63             | OK     |
| 128     | 8192    | 3.4436       | 7.4871       | 4401.00             | OK     |
| 128     | 16384   |              |              |                     | OOM    |

This was benchmarked on an A100 with 40GB RAM.
At smaller sequence lengths, the memory after forward grows linearly with sequence length, but at larger sequence lengths, the memory after forward grows much faster. So the relationship is likely O(seq_len^2). One way to eliminate this memory cost is to recompute the attention activations during backward instead of storing it, trading off extra compute for saved memory.

# Problem (torch_compile)

(a)
Benchmarked on A100 with 40GB RAM (same as above).

| d_model | seq_len | baseline fwd (s) | compile fwd (s) | baseline bwd (s) | compile bwd (s) | compiled mem after fwd (MB) |
|---------|---------|------------------|------------------|------------------|------------------|-------------------------------|
| 16      | 256     | 0.0325           | 0.0241           | 0.0608           | 0.0468           | 21.22                          |
| 16      | 1024    | 0.0725           | 0.0491           | 0.1672           | 0.1078           | 84.88                          |
| 16      | 4096    | 0.7568           | 0.4390           | 1.8534           | 1.0870           | 1070.75                        |
| 16      | 8192    | 2.8289           | 1.7051           | 6.8483           | 3.8852           | 4205.25                        |
| 16      | 16384   | OOM              | 5.7394           | OOM              | 15.5994          | 16714.25                       |
| 32      | 256     | 0.0309           | 0.0275           | 0.0621           | 0.0417           | 22.09                          |
| 32      | 1024    | 0.0652           | 0.0485           | 0.1617           | 0.1089           | 88.38                          |
| 32      | 4096    | 0.7833           | 0.5438           | 1.8849           | 1.1057           | 1084.75                        |
| 32      | 8192    | 2.9185           | 1.9516           | 6.9421           | 3.9803           | 4233.25                        |
| 32      | 16384   | OOM              | 6.2884           | OOM              | 15.9829          | 16770.25                       |
| 64      | 256     | 0.0309           | 0.0285           | 0.0621           | 0.0434           | 23.84                          |
| 64      | 1024    | 0.0682           | 0.0701           | 0.1657           | 0.1128           | 95.38                          |
| 64      | 4096    | 0.8284           | 0.5183           | 1.9336           | 1.1511           | 1112.75                        |
| 64      | 8192    | 3.1000           | 1.7442           | 7.1270           | 4.0755           | 4289.25                        |
| 64      | 16384   | OOM              | 7.0036           | OOM              | 16.7186          | 16882.25                       |
| 128     | 256     | 0.0309           | 0.0287           | 0.0582           | 0.0456           | 27.34                          |
| 128     | 1024    | 0.0741           | 0.0763           | 0.1726           | 0.1218           | 109.38                         |
| 128     | 4096    | 0.9141           | 0.6079           | 2.0257           | 1.2495           | 1168.75                        |
| 128     | 8192    | 3.4436           | 2.0895           | 7.4871           | 4.4394           | 4401.25                        |
| 128     | 16384   | OOM              | 8.3827           | OOM              | 18.1259          | 17106.25                       |


(b)

Full end-to-end training step (forward + backward + optimizer step), baseline vs. torch.compile

| d_model | context_length | baseline mean_time (s) | compiled mean_time (s) |
|---------|-----------------|-------------------------|--------------------------|
| 768     | 128             | 0.072913                | 0.064861                 |
| 768     | 256             | 0.120324                | 0.100844                 |
| 768     | 512             | 0.241405                | 0.189355                 |
| 1024    | 128             | 0.219400                | 0.201784                 |
| 1024    | 256             | 0.369397                | 0.320342                 |
| 1024    | 512             | 0.723469                | 0.581659                 |
| 1280    | 128             | 0.518490                | 0.487466                 |
| 1280    | 256             | OOM                     | 0.750421                 |
| 1280    | 512             | OOM                     | OOM                       |
| 1600    | 128             | OOM                     | OOM                       |
| 1600    | 256             | OOM                     | OOM                       |
| 1600    | 512             | OOM                     | OOM                       |
| 2560    | 128             | OOM                     | OOM                       |
| 2560    | 256             | OOM                     | OOM                       |
| 2560    | 512             | OOM                     | OOM                       |


