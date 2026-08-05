---
layout: page
permalink: /assignments/assignment1
title: "Assignment 1: Kernels and Compilation"
---
#### **Released:** TBD <br/> **Due:** TBD
{: .no_toc}

* (The list will be replaced with the table of contents.)
{:toc}

### Part 0: Overview and Setup

#### Overview

In this assignment you will build the attention kernel at the heart of every
modern LLM. First you will run the naive attention implementation
that materializes the full attention matrix, then you will implement Flash Attention: a tiled CUDA kernel that computes exactly the same result without ever materializing the `N x N` score matrix, using the online softmax trick.
You will compile your kernel with `nvcc`, integrate it into **PyTorch** as a CUDA extension with a custom `autograd.Function`, and use it to train a real Transformer language model on German-to-English machine translation. Finally, you will implement the same algorithm a second time in **Triton**, where the matmuls run on tensor cores — and measure it beating naive attention by an order of magnitude.

After this assignment you should be able to:
* Explain why attention is memory-bound and how tiling against the GPU memory hierarchy (HBM vs. on-chip shared memory) addresses it.
* Derive and implement the online (streaming) softmax update, including numerically stable max-subtraction.
* Implement causal masking inside a tiled kernel, where thread-local tile coordinates must be translated to global sequence positions.
* Compile a CUDA kernel and integrate it into PyTorch as a CUDA extension backing a custom `autograd.Function` — the standard way custom ops ship in real systems.
* Measure a kernel honestly (CUDA events, peak-memory accounting) and reason about when tiling buys memory, when it buys speed, and why the two are different questions.
* Write a block-level kernel in Triton and explain what tensor cores change about kernel design.

This assignment is completed **individually** on **Google Colab**. A free T4 GPU is sufficient. (You may use any CUDA GPU you have access to instead; everything is plain `nvcc` + Python.)

#### Setup

1. **Get the skeleton code.** Create a **private** repository from the template at <TODO: GITHUB REPO TEMPLATE>
   (`Use this template > Create a new repository`, select `Private`).
2. **Open the Colab notebook.** Go to [colab.research.google.com](https://colab.research.google.com) → `File > Open notebook` → `GitHub` tab → check *Include private repos* and authorize Colab's GitHub app → select your repository → `flash_attention_colab.ipynb`. Set the runtime to a GPU: `Runtime > Change runtime type > T4 GPU`. (Opening a notebook from GitHub loads only the notebook itself into a fresh VM — the notebook's clone cell fetches the rest of your repository, which is where your edits will live.)
3. **Run the setup cells.** The first cells of the notebook clone your
   repository (edit `REPO_URL` to point at your private repo), install the few missing dependencies, and compile your kernel for the standalone graded tests:
   ```
   bash compile_cuda.sh
   ```
   This produces shared libraries under `build/` that the `ctypes`-based
   graded tests load. The PyTorch extension used in the later cells
   compiles separately — and automatically — on first use.

**Note**: Colab VMs are ephemeral. **Commit and push your changes to your private repository at the end of every session**, or you will lose them.
The notebook edits files in the cloned repo on the VM; use the provided `git` cells (or the Colab terminal) to push.

**Note**: If a CUDA error leaves the GPU in a bad state (`CUDA error: an illegal memory access was encountered` on every subsequent call), restart the runtime: `Runtime > Restart session`, then re-run the compile cell.

#### Codebase tour

You will only modify **two files**, but you should understand how they
connect to the rest of the stack:

| Path | What it is |
|---|---|
| `src/flash_attention_kernel.cu` | The Flash Attention forward kernel and its `extern "C"` launcher. Contains the three `TODO` blocks you implement. |
| `src/flash_attention_torch.cu` | PyTorch extension wrapper: `#include`s your kernel file and launches your `__global__` kernel directly on GPU-resident torch tensors — no host copies. |
| `flash_attention.py` | JIT-compiles the extension (`torch.utils.cpp_extension`) and wraps it in a `torch.autograd.Function`. The backward pass is provided (it recomputes standard attention gradients with torch ops), so the model can train even though you only write the forward kernel. Also provides the `naive_attention` baseline. |
| `triton_flash_attention.py` | Part 4: the same algorithm in Triton, on tensor cores. Contains the one `TODO` block you implement (the online-softmax merge). | 
| `kernel_tests/` | Graded tests (see [How to test](#how-to-test)). |
| `project/run_machine_translation.py` | Trains a 4-layer decoder-only Transformer on IWSLT-14 De-En; `--use_flash_attn True` swaps the attention core from naive to your kernel. |

### Part 1: Background

Our kernel follows the [FlashAttention](https://arxiv.org/abs/2205.14135) paper's notation (`Br`, `Bc`, `l`, `m`).

#### 1.1 Attention, and why the naive version is memory-bound

Given query, key, and value matrices `Q, K, V` of shape `(N, d)` per head
(`N` = sequence length, `d` = head dimension), attention computes:

```
S = Q @ K^T / sqrt(d)        # (N, N) scores
P = softmax(S, dim=-1)       # (N, N) probabilities
O = P @ V                    # (N, d) output
```

The naive implementation writes the `(N, N)` matrices `S` and `P` to GPU
main memory (HBM) and reads them back. For `N = 4096`, one head's score
matrix is 64 MB — and every byte of it is written once and read once, while
the useful inputs/outputs (`Q`, `K`, `V`, `O`) are only `4 * N * d` floats.
The arithmetic is cheap; the HBM traffic is the bottleneck. GPUs have a
small (tens of KB per SM) but ~10x-faster on-chip **shared memory (SRAM)**;
the entire Flash Attention idea is: *tile the computation so `S` and `P`
only ever exist one small tile at a time, in SRAM*.

#### 1.2 Numerically stable softmax

`softmax(x)_i = exp(x_i) / sum_j exp(x_j)` overflows for scores as small as
89 in float32. The standard fix subtracts the row max `m = max_j x_j`
before exponentiating (this cancels in the ratio):

```
softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)
```

Note the two per-row statistics: the max `m` and the normalizer
`l = sum_j exp(x_j - m)`.

#### 1.3 Online softmax: merging partial results

The blocker for tiling is that softmax's `m` and `l` are *row-global*: they
depend on all `N` scores in a row, but a tile only sees `Bc` of them. The
key insight (Milakov & Gimelshein 2018, used by Flash Attention) is that
`(m, l)` — and the partial output `O` — can be **updated incrementally** as
new tiles arrive. If the running statistics after previous tiles are
`(m_prev, l_prev, O_prev)` and the current tile alone yields
`(m_tile, l_tile, PV_tile)` where `PV_tile = exp(S_tile - m_tile) @ V_tile`,
then the merged statistics are:

```
m_new = max(m_prev, m_tile)
l_new = exp(m_prev - m_new) * l_prev  +  exp(m_tile - m_new) * l_tile
O_new = ( l_prev * exp(m_prev - m_new) * O_prev  +  exp(m_tile - m_new) * PV_tile ) / l_new
```

The correction factors `exp(m_prev - m_new)` and `exp(m_tile - m_new)`
re-express previously computed exponentials relative to the new max. After
the last tile, `O` equals exact attention — no approximation anywhere.
Convince yourself of this before writing code; it is the entire assignment.

#### 1.4 The kernel structure (given to you)

The skeleton in `src/flash_attention_kernel.cu` already implements the
tiling scaffold, following Algorithm 1 of the paper:

* One thread block per `(batch, head)`; `Bc = Br = 32` threads per block,
  one thread per query row of a tile.
* Outer loop over K/V tiles `j`: each iteration loads one `(Bc, d)` tile of
  `K` and `V` from HBM into shared memory.
* Inner loop over Q tiles `i`: loads a `(Br, d)` tile of `Q`, and reads the
  running `m`, `l` for its rows from HBM.
* **Your code** computes the tile's scores, its local softmax statistics,
  and performs the online merge into `O`, `m`, `l` (Section 1.3).

Because the block size is fixed at 32, the kernel requires **`seq_len` to
be a multiple of 32**, and the shared-memory budget requires
**`head_dim <= 64`** (this keeps `3*Bc*d + Bc*Br` floats under the 48 KB
per-block limit of the Colab T4). The graded tests respect both limits.

#### 1.5 Causal masking

A decoder LM must not attend to future positions: score `S[q, k]` is valid
only when `k <= q` (global positions). Inside the kernel you work in tile
coordinates, so the check must be translated: thread `tx` in Q-tile `i`
handles global row `i*Br + tx`, and column `y` of K-tile `j` is global
column `j*Bc + y`. Masked entries are set to `-inf` *before* the max, so
they contribute `exp(-inf) = 0` to `l` and `O`. The diagonal (`k == q`) is
always valid, so the row max stays finite. (The scaffold also gives you an
already-implemented *tile-level* skip: if an entire K-tile is in a row's
future, the iteration is skipped outright.)

### Part 2: Implement the Flash Attention kernel

All work is inside `forward_kernel` in `src/flash_attention_kernel.cu`,
in the three marked blocks. Implement them in order — each is tested
cumulatively by the graded tests.

#### 2.1 `ASSIGN5_1_1`: tile scores, causal mask, and row max

For your thread's query row, compute the `Bc` dot-product scores against
the K-tile in shared memory, scaled by `1/sqrt(d)`; apply the causal mask
(Section 1.5) when `causal` is set; store the scores into the shared
buffer `S`; and track the row max `row_m`.

#### 2.2 `ASSIGN5_1_2`: tile softmax numerator and row sum

Overwrite each stored score with `exp(score - row_m)` (use `__expf`) and
accumulate the row sum `row_l`.

#### 2.3 `ASSIGN5_1_3`: the online softmax merge

This is the heart of the assignment. Read the running `(m_prev, l_prev)`
(already loaded for you), compute the merged `(m_new, l_new)` per Section
1.3, rescale-and-accumulate the output rows
(`P_tile @ V_tile` against `O_prev`), and write `O`, `m`, `l` back to HBM.
The exact update formulas are restated in the comment above the block.

**Hints**
* Every array index you need already appears in working form elsewhere in
  the kernel (the Q/K/V loads, the `lm_offset` reads). Mimic them.
* The two `__syncthreads()` in the scaffold are the only synchronization
  required — your blocks are purely thread-local except for reads of the
  shared `Kj`, `Vj`, `S` tiles.
* If your non-causal output is correct but causal fails only for
  `N > 32`, your mask is comparing tile-local instead of global positions.
* If results are correct for `N = 32` but wrong for larger `N`, your merge
  is not order-independent — recheck the correction factors.

### Part 3: Measure your kernel and use it in a real Transformer

Nothing to implement here — this part measures what your kernel does and
does not buy, and demonstrates it as a drop-in replacement inside a full
model. All of it goes in your report.

1. **Speed benchmark** (notebook cell 7): times your kernel against the
   naive baseline with CUDA events, GPU-resident, at several sequence
   lengths. **Your kernel will be slower — that is the expected result.**
   The naive baseline's two matmuls run on cuBLAS (tensor cores, carefully
   tiled by NVIDIA engineers), while your kernel's inner products are
   deliberately simple scalar loops so that the algorithm stays readable.
   Production Flash Attention wins on *both* memory and speed because it
   fuses the online-softmax algorithm you wrote with matmuls of cuBLAS
   quality. You will explain this in Q3.
2. **Memory measurement** (notebook cell 8): measures peak extra GPU
   memory for one attention call at `N = 2048`: roughly **3 GiB (naive)
   vs. 50 MiB (yours)** — a ~62x reduction that grows linearly with `N`.
   This is the half of the Flash Attention promise your kernel fully
   delivers: the `(N, N)` matrix never exists. Push `N` higher and the
   naive baseline exhausts a T4's 16 GB while your kernel keeps going.
3. **Machine translation training** (notebook cells 12-13): trains the
   4-layer decoder LM on IWSLT-14 De-En, first with naive attention, then
   with your flash kernel:
   ```
   python project/run_machine_translation.py --model_max_length 65 \
       --samples_per_epoch 20000 --batch_size 64 [--use_flash_attn True]
   ```
   (`model_max_length 65` makes the model sequence length 64, a multiple
   of 32 as your kernel requires.) Capture the final progress-bar line of
   both runs. The loss values should match almost exactly — your kernel
   computes *exact* attention, and gradients flow through the provided
   backward — making this a strong end-to-end correctness check. Expect
   `tokens_per_sec` to be modestly lower with your kernel (attention is a
   small fraction of this short-sequence model; Amdahl's law caps the
   impact of any single op either way).

### Part 4: Make it fast — the same algorithm in Triton

Part 3 left you with a puzzle: your kernel does *less* memory traffic than
the naive baseline yet runs slower, because its inner products are scalar
loops on CUDA cores while cuBLAS uses **tensor cores** — dedicated matrix
units that are an order of magnitude faster at matmuls. Hand-writing
tensor-core code (`wmma`/CUTLASS, what production Flash Attention uses) is
beyond a first kernel course. But there is an industry-standard middle
ground: [Triton](https://triton-lang.org), a Python-embedded DSL where you
write **block-level** programs — "load this `(64, 64)` tile, `tl.dot`
these tiles" — and the compiler emits tensor-core instructions, memory
pipelining, and register allocation for you. (This is not a toy:
`torch.compile` generates Triton, and much industrial kernel work is
prototyped in it.)

`triton_flash_attention.py` contains the full kernel scaffold: block
pointers, the Q-block load, the K/V loop, tile scores via `tl.dot`, and
causal masking are all provided. Notice how it differs from your CUDA
kernel: one *program instance* per `(Q-block, batch, head)` — thousands of
independent instances instead of one warp per `(batch, head)` — and every
operation acts on whole `(64, 64)` tiles.

#### 4.1 `ASSIGN5_1_4`: the online softmax merge, on tiles

Implement the one `TODO` block: the same merge you wrote in `ASSIGN5_1_3`,
now expressed with Triton tile primitives (`tl.maximum`, `tl.max`,
`tl.exp`, `tl.sum`, `tl.dot`). The comment above the block gives the
required quantities in order; it is ~6 lines. Two things to notice while
you write it:
* The running statistics `m_i`, `l_i`, `acc` are `float32` while the tile
  data is `float16` — mixed precision, exactly as in production Flash
  Attention (cast `p` with `.to(v.dtype)` before the `tl.dot`).
* There is no explicit `O` read-modify-write against HBM as in your CUDA
  kernel: `acc` lives in registers for the whole loop and is written once.

Then run the benchmark (notebook cell 10). At `N = 4096` you should see
your Triton kernel beat naive attention by **an order of magnitude** —
same algorithm you wrote in CUDA, now with matmuls the hardware likes.

### How to test

All commands run from the repo root (the notebook cells do this for you).
**After every kernel edit, recompile**: run `bash compile_cuda.sh` for the
`ctypes` tests; the PyTorch extension detects source changes and recompiles
itself on the next call.

* **Graded kernel test (non-causal):**
  ```
  python kernel_tests/grade_flash_kernel.py --causal 0
  ```
  Compares your kernel against a NumPy reference at four shapes up to
  `N = 512`, tolerance `1e-3` (typical achieved error is `~1e-6`). Prints
  `PASS`/`FAIL` per shape and exits nonzero on any failure.
* **Graded kernel test (causal):** the same with `--causal 1`.
* **PyTorch integration test:**
  ```
  python kernel_tests/test_flash_attn_torch.py
  ```
  Runs your kernel through the extension and the `autograd.Function`,
  checking the forward against `torch.nn.functional.scaled_dot_product_attention`
  and gradients against torch autograd, causal and non-causal. The first
  run JIT-compiles the extension (~1-2 minutes); later runs hit a cache.
* **Graded Triton test (Part 4):**
  ```
  python kernel_tests/grade_triton_flash.py
  ```
  Checks your Triton kernel against
  `torch.nn.functional.scaled_dot_product_attention` in float16, causal
  and non-causal. Triton also JIT-compiles on first call (seconds).
* **Debugging tips:**
  * Test `ASSIGN5_1_1`/`5_1_2` before writing `5_1_3`: with only the first
    two blocks done, the `N = 32` case (a single tile) exercises them in
    isolation — a single-tile run never needs the merge (with `l/m`
    initialized as given, the first merge reduces to a plain write).
  * `printf` works inside CUDA kernels and is your friend; guard it with
    `if (bx == 0 && by == 0 && tx == 0)` to avoid drowning in output.
  * The reference implementation in `kernel_tests/grade_flash_kernel.py`
    (`reference()`, ~10 lines of NumPy) is the ground truth — port it to a
    scratch cell and compare intermediate quantities if you are stuck.

### Submission

Run the **Package submission** notebook cell, which creates
`submission.zip` containing exactly:
* `src/flash_attention_kernel.cu`
* `triton_flash_attention.py`

Submit to the Canvas Assignments page:
1. `submission.zip`, and
2. a **PDF report** (any tool, but submit PDF) containing:
   * your name and EID;
   * the output of both graded kernel tests and the integration test;
   * the Part 3 and Part 4 evidence: the CUDA benchmark table (cell 7),
     the peak-memory numbers (cell 8), the Triton benchmark table
     (cell 10), and the final training line for the naive and flash MT
     runs (cells 12-13);
   * **Q1**: For `batch=8, heads=12, N=2048` (the cell 8 configuration),
     derive the ~3 GiB and ~50 MiB peak-memory numbers you measured. What
     does each consist of? Show the arithmetic.
   * **Q2**: Your kernel processes K/V tiles sequentially, merging each
     into the running `(m, l, O)`. Explain in 3-5 sentences why the merged
     result is independent of the order in which tiles are processed, and
     name one system opportunity this order-independence enables.
   * **Q3**: You implemented the same algorithm twice. Your CUDA kernel is
     slower than naive attention (cell 7) while your Triton kernel is an
     order of magnitude faster (cell 10) — a gap of roughly three orders
     of magnitude between your own two implementations. Using your
     measurements, explain where the gap comes from. Be specific about
     (a) the hardware units each implementation's matmuls run on, (b) the
     parallelization difference (how many independent blocks/programs each
     launches), and (c) why the naive baseline, despite cuBLAS, still
     loses to your Triton kernel at large `N`.

### Grading

* CUDA kernel (Part 2): **55%**
  * Forward correctness, non-causal (`grade_flash_kernel.py --causal 0`): 30%
  * Causal masking (`grade_flash_kernel.py --causal 1`): 15%
  * PyTorch integration (`test_flash_attn_torch.py`): 10%
* Triton kernel (Part 4, `grade_triton_flash.py`): **15%**
* Report: **30%**
  * Test outputs and Part 3/4 evidence (benchmarks, memory, both training runs): 12%
  * Q1: 6%
  * Q2: 6%
  * Q3: 6%

Kernel scores are all-or-nothing per test (the tests exit nonzero on any
failing shape), and we grade with a fresh compile of your two submitted
files against unmodified course code — do not modify files outside
`src/flash_attention_kernel.cu` and `triton_flash_attention.py`.

### FAQ / Common issues

* **`cannot open shared object file` when running a graded test** — you
  skipped the compile cell (or the runtime restarted). Run
  `bash compile_cuda.sh`.
* **I edited the kernel but the integration test result did not change** —
  the extension recompiles automatically on source changes, but the
  `ctypes` tests load `build/*.so`: rerun `bash compile_cuda.sh`. If in
  doubt, recompile both.
* **The first extension build takes minutes / prints a wall of compiler
  output** — normal; it is cached afterwards.
* **Everything fails after one bad run** — an illegal memory access poisons
  the CUDA context. `Runtime > Restart session`, recompile, rerun.
* **`head_dim too large for shared memory`** — the graded shapes keep
  `d <= 64`, so this usually indicates a broken index computation writing
  out of bounds rather than a real limit problem.
* **Causal test wrong only in the last rows/columns** — off-by-one in the
  global-position translation; remember the diagonal is *kept*.
* **My CUDA kernel is slower than naive attention** — expected; that is
  the point of Part 4 and Q3.
* **My Triton kernel returns NaN** — with the merge block empty (or
  wrong), `l_i` stays 0 and the final division is 0/0. If your merge looks
  right, check that `alpha` multiplies *both* `l_i` and `acc`.
* **`tl.dot` complains about dtypes** — cast the probabilities to the
  value dtype before the matmul: `tl.dot(p.to(v.dtype), v)`.

### Acknowledgments

This assignment builds on
[Assignment 4 of 11-868 *Large Language Model Systems*](https://llmsystem.github.io/llmsystemhomework/assignment_4/)
at Carnegie Mellon University — the custom-CUDA-kernel workflow and the
machine translation project are adapted from that course's assignment
series. We thank the 11-868 instructors for making their course materials
publicly available.
