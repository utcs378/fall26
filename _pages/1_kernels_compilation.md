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
You will compile your kernel with `nvcc`, bind it into the **miniTorch** deep-learning framework, and use it to train a real Transformer language model on German-to-English machine translation.

After this assignment you should be able to:
* Explain why attention is memory-bound and how tiling against the GPU memory hierarchy (HBM vs. on-chip shared memory) addresses it.
* Derive and implement the online (streaming) softmax update, including numerically stable max-subtraction.
* Implement causal masking inside a tiled kernel, where thread-local tile coordinates must be translated to global sequence positions.
* Compile a CUDA kernel into a shared library and call it from Python through a host-pointer `ctypes` interface.

This assignment is completed **individually** on **Google Colab**. A free T4 GPU is sufficient. (You may use any CUDA GPU you have access to instead; everything is plain `nvcc` + Python.)

#### Setup

1. **Get the skeleton code.** Create a **private** repository from the template at <TODO: GITHUB REPO TEMPLATE>
   (`Use this template > Create a new repository`, select `Private`).
2. **Open the Colab notebook.** Open `flash_attention_colab.ipynb` from your repository in [Google Colab](https://colab.research.google.com), and set the runtime to a GPU: `Runtime > Change runtime type > T4 GPU`.
3. **Run the setup cells.** The first cells of the notebook clone your
   repository (edit `REPO_URL` to point at your private repo), install the one missing dependency (`pycuda`), and compile all CUDA kernels:
   ```
   bash compile_cuda.sh
   ```
   Compilation must succeed before any test can run. It compiles each
   `src/*.cu` file into a shared library under `minitorch/cuda_kernels/`.

**Note**: Colab VMs are ephemeral. **Commit and push your changes to your private repository at the end of every session**, or you will lose them.
The notebook edits files in the cloned repo on the VM; use the provided `git` cells (or the Colab terminal) to push.

**Note**: If a CUDA error leaves the GPU in a bad state (`CUDA error: an illegal memory access was encountered` on every subsequent call), restart the runtime: `Runtime > Restart session`, then re-run the compile cell.

#### Codebase tour

You will only modify **one file**, but you should understand how it connects
to the rest of the stack:

| Path | What it is | Modify? |
|---|---|---|
| `src/flash_attention_kernel.cu` | The Flash Attention forward kernel and its `extern "C"` launcher. Contains the three `TODO` blocks you implement. | **Yes** |
| `minitorch/cuda_kernel_ops.py` | `flash_attn_fw`: the `ctypes` binding that calls your compiled kernel from Python. | No |
| `minitorch/tensor_functions.py` | `FlashAttn(Function)`: hooks your kernel into miniTorch autograd. The backward pass is provided (it recomputes standard attention gradients), so the model can *train* even though you only write the forward kernel. | No |
| `minitorch/modules_transfomer.py` | `MultiHeadAttention(use_flash_attn=True)`: the Transformer uses your kernel when this flag is set. | No |
| `kernel_tests/` | Correctness tests (see [How to test](#how-to-test)). | No |
| `project/run_machine_translation.py` | Trains a 4-layer decoder-only Transformer on IWSLT-14 De-En. | No |

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

### Part 3: Use your kernel in a real Transformer

Nothing to implement here — this part demonstrates that your kernel is a
drop-in replacement inside a full model, and it is part of your report.

1. **Module equivalence** (notebook cell 6): runs miniTorch
   `MultiHeadAttention` twice with identical weights — naive path vs.
   `use_flash_attn=True` — and prints the max forward/gradient difference.
   Both should be `< 1e-5`.
2. **Machine translation training** (notebook cells 7-8): trains the
   4-layer decoder LM on IWSLT-14 De-En for one epoch, first with naive
   attention, then with your flash kernel:
   ```
   python project/run_machine_translation.py --model_max_length 65 \
       --samples_per_epoch 1280 --batch_size 64 [--use_flash_attn True]
   ```
   (`model_max_length 65` makes the model sequence length 64, a multiple
   of 32 as your kernel requires.) Capture the final progress-bar line of
   both runs (loss and `tokens_per_sec`) for your report. The two loss
   curves should track each other closely — your kernel computes *exact*
   attention, so this is a strong end-to-end correctness check.

**A note on speed.** Do not expect a wall-clock speedup in this harness:
the teaching framework round-trips every tensor through host memory, which
dominates runtime, and this kernel's inner matmuls are deliberately simple.
What you should expect — and what you will quantify in the report — is the
*memory* difference: naive attention materializes `batch * heads * N * N`
scores; your kernel's HBM footprint for the same quantities is
`O(batch * heads * N)` (just `l` and `m`).

### How to test

All commands run from the repo root with `PYTHONPATH=.` (the notebook cells
do this for you). **Recompile after every kernel edit** (`bash
compile_cuda.sh` or the notebook's compile cell) — Python loads the `.so`,
not your source.

* **Graded kernel test (non-causal):**
  ```
  python kernel_tests/grade_flash_kernel.py --causal 0
  ```
  Compares your kernel against a NumPy reference at four shapes up to
  `N = 512`, tolerance `1e-3` (typical achieved error is `~1e-6`). Prints
  `PASS`/`FAIL` per shape and exits nonzero on any failure.
* **Graded kernel test (causal):** the same with `--causal 1`.
* **miniTorch integration test:**
  ```
  python kernel_tests/test_flash_attn_fw.py
  ```
  Exercises the full `Tensor.flash_attn` path through the framework.
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

Run the final notebook cell (**Package submission**), which creates
`submission.zip` containing exactly:
* `src/flash_attention_kernel.cu`

Submit to the Canvas Assignments page:
1. `submission.zip`, and
2. a **PDF report** (any tool, but submit PDF) containing:
   * your name and EID;
   * the output of both graded kernel tests and the integration test;
   * the Part 3 screenshots: module-equivalence output, and the final
     training line for the naive and flash MT runs;
   * **Q1**: For `batch=64, heads=8, N=64` (the MT configuration), how many
     floats does naive attention materialize for `S`, and what is your
     kernel's corresponding HBM-resident footprint (`l` and `m`)? Show the
     arithmetic.
   * **Q2**: Your kernel processes K/V tiles sequentially, merging each
     into the running `(m, l, O)`. Explain in 3-5 sentences why the merged
     result is independent of the order in which tiles are processed, and
     name one system opportunity this order-independence enables.

### Grading

* Flash Attention kernel implementation: **70%**
  * Forward correctness, non-causal (`grade_flash_kernel.py --causal 0`): 40%
  * Causal masking (`grade_flash_kernel.py --causal 1`): 20%
  * miniTorch integration (`test_flash_attn_fw.py`): 10%
* Report: **30%**
  * Test outputs and Part 3 evidence (equivalence + both training runs): 15%
  * Q1: 7.5%
  * Q2: 7.5%

Kernel scores are all-or-nothing per test (the tests exit nonzero on any
failing shape), and we grade with a fresh compile of your submitted
`flash_attention_kernel.cu` against unmodified course code — do not modify
files outside `src/flash_attention_kernel.cu`.

### FAQ / Common issues

* **`cannot open shared object file` when running a test** — you skipped
  the compile cell (or the runtime restarted). Run `bash compile_cuda.sh`.
* **Everything fails after one bad run** — an illegal memory access poisons
  the CUDA context. `Runtime > Restart session`, recompile, rerun.
* **`requested shared memory exceeds device limit`** — your `head_dim` is
  too large; the graded shapes keep `d <= 64`, so this indicates a broken
  index computation writing out of bounds rather than a real limit problem.
* **Causal test wrong only in the last rows/columns** — off-by-one in the
  global-position translation; remember the diagonal is *kept*.
* **My kernel is slower than naive attention** — expected in this harness;
  see the note in Part 3.

### Acknowledgments

This assignment builds on
[Assignment 4 of 11-868 *Large Language Model Systems*](https://llmsystem.github.io/llmsystemhomework/assignment_4/)
at Carnegie Mellon University — the miniTorch-based framework, the CUDA
kernel/`ctypes` integration workflow, and the machine translation project
are adapted from that course's assignment series. We thank the 11-868
instructors for making their course materials publicly available.
miniTorch itself is by Sasha Rush and collaborators
([minitorch.github.io](https://minitorch.github.io)).
