---
layout: page
permalink: /assignments/assignment1
title: "Assignment 1: Kernels and Compilation"
---

#### **Released:** 09/10/2026 <br/> **Due:** 09/23/2026, 11:59 PM CT
{: .no_toc}

* (The list will be replaced with the table of contents.)
{:toc}

### Part 0: Overview and Setup

#### Overview

In this assignment you will build the attention kernel at the heart of every
modern LLM, and then reason about how its design must change with context
length and block size. You start from a naive attention implementation that
materializes the full `N x N` score matrix in GPU main memory, fuse it into a
single tiled CUDA kernel that never writes `S` or `P` to HBM (Flash Attention,
via the online-softmax trick), and finish by analyzing — on paper and
empirically — how the tile sizes `Br x Bc` should be chosen as a function of
sequence length `N` and head dimension `d`.

After this assignment you should be able to:
* Explain why attention is memory-bound and how tiling against the GPU memory
  hierarchy (HBM vs. on-chip shared memory) addresses it.
* Derive and implement the online (streaming) softmax update, including
  numerically stable max-subtraction.
* Fuse the score, softmax, and output matmuls into one kernel that keeps every
  intermediate tile in SRAM.
* Derive a kernel's shared-memory budget and reason about the occupancy
  trade-off that caps the block size.
* Measure a kernel honestly (CUDA events, HBM-traffic accounting) and explain
  when a given tile size is limited by memory, by occupancy, or by matmul
  throughput.

This assignment is to be completed with your **registered group** (check your
group number [here](https://docs.google.com/spreadsheets/d/1gqZPBOdgAd2ViVLmAQw8IVMt3Q2tH4bmN8zG5BI9s7k/edit?usp=sharing){:target="_blank"}).
You will need access to a GPU equipped instance for this project. You can use 
`gpulease.py` to request resources for your group.
Each group is allocated **20 GPU hours** for assignment 1, and you are
responsible for completing the assignment within this budget.
You can use the script to stop your instances while you aren't using them, but
**beware** this will delete all the files on your instances, so be
sure to save your progress elsewhere before doing so.

#### Setup

1. **Get the skeleton code.** Create a **private** repository from the
   [assignment 1 template repository](https://github.com/utcs378/assignment1-template){:target="_blank"}
   (`Use this template > Create a new repository`, select `Private`).
2. **Lease a GPU node.** Clone your repository to your own machine, then run
   `resource_request/gpulease.py` from the repository root to bring up your
   group's node — an AWS `g4dn.xlarge` with one NVIDIA T4:
   ```bash
   python3 resource_request/gpulease.py login <your-token>  # once per machine
   python3 resource_request/gpulease.py start               # brings the node up
   python3 resource_request/gpulease.py status              # ssh command, budget
   ```
   Read `resource_request/README.md` before your first session: it covers the
   20 GPU-hour budget, connecting from VS Code, and why you must run
   `gpulease.py stop` as soon as you are done. Clone your repository onto the
   node too, once you are connected.
3. **Create a virtual environment and install dependencies.** On the node,
   from `attn_kernel_assignment/` in your clone:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Re-run `source .venv/bin/activate` in **every new shell** before running the
   tests, or they will use the system Python.
4. **Compile your kernels** for the standalone graded tests, from that same
   directory:
   ```bash
   bash compile_cuda.sh
   ```

**Note**: `stop`, the end of your lease, and the assignment deadline all destroy
the node **and its disk** — there is no snapshot and no undo. **Commit and push
at the end of every session** or you will lose your work.

**Note**: If a CUDA error leaves the GPU in a bad state
(`CUDA error: an illegal memory access was encountered` on every subsequent
call), exit the Python process — the poisoned CUDA context dies with it — and
rerun. You do not need to recompile unless you edited a kernel.

#### Codebase tour

You will modify **three files**; understand how they connect to the rest of the
stack:

| Path | What it is |
|---|---|
| `reference/attention_ref.py` | The PyTorch/NumPy reference and the test harness (Part 1). You fill in the reference; the harness is used by every later part. |
| `src/naive_attention_kernel.cu` | The three-kernel baseline (Part 2): `S = QK^T`, row-wise softmax, `O = PV`, materialized in HBM. Contains two `TODO` blocks. |
| `src/flash_attention_kernel.cu` | The fused Flash Attention forward kernel (Part 3). Contains the tiling scaffold and three `TODO` blocks. |
| `kernel_tests/` | Graded tests and benchmark drivers. |
| `bench/sweep.py` | The block-size / sequence-length sweep driver used in Part 4c. |

You will **not** modify anything under `kernel_tests/` or `bench/`; we grade
with a fresh compile of your three files against unmodified course code.

### Part 1: Reference implementation and test harness

Before writing any CUDA, provide the ground truth everything else is checked
against.

In `reference/attention_ref.py`, implement `attention_reference(Q, K, V,
causal=False)` in PyTorch or NumPy computing `softmax(Q @ K^T / sqrt(d)) @ V`,
and complete `check_against_reference(kernel_out, Q, K, V, ...)`, which compares
a kernel's output to the reference and returns `PASS`/`FAIL`. Use a tolerance of
`1e-2` for fp16 and `1e-4` for fp32. Assume no dropout and no attention mask
unless a test sets `causal=True`.

**Deliverables.** The completed reference and harness; every subsequent part
imports and calls `check_against_reference`. Confirm it passes trivially when
fed the reference's own output.

### Part 2: Baseline — attention materialized in HBM

Implement attention as **three separate CUDA kernels** in
`src/naive_attention_kernel.cu`. Provide a plain (non-fused, non-tiled) matmul
and a row-wise softmax; do not use cuBLAS. Write the intermediates to HBM:

```text
1. S = Q @ K^T * scale        # (N, N), written to HBM
2. P = softmax(S, dim=-1)      # row-wise, numerically stable (subtract row max)
3. O = P @ V                   # (N, d)
```

This is your **baseline** — the point of comparison for the fused kernel, not a
performance target. Materializing `S` and `P` costs `O(N^2)` HBM traffic and
`O(N^2)` memory; for `N = 4096` one head's `S` is 64 MB and every byte is
written once and read once, while the useful data (`Q, K, V, O`) is only
`4 * N * d` floats. The arithmetic is cheap; the HBM traffic dominates.

The two `TODO` blocks are `ASSIGN1_2_1` (the `QK^T` scores) and `ASSIGN1_2_2`
(the `PV` output); the stable row-wise softmax between them is provided.

**Deliverables.**
* Baseline passing the Part 1 harness (`kernel_tests/grade_naive.py`).
* A table of runtime and **HBM bytes moved** for `N in {512, 1024, 2048,
  4096}`, `d = 64`. Calculate the `N` at which you run out of memory.

### Part 3: Fuse into a Flash Attention kernel

Fuse Parts 1 and 2 into a **single** kernel in
`src/flash_attention_kernel.cu`. Tile `Q` into blocks of `Br` rows and `K, V`
into blocks of `Bc` columns. For each `Q` tile, loop over the `K/V` tiles,
compute the partial scores in SRAM, and update the output accumulator with the
online-softmax correction. **Never write `S` or `P` to HBM.**

We follow the [FlashAttention](https://arxiv.org/abs/2205.14135){:target="_blank"} paper's
notation (`Br`, `Bc`, `l`, `m`).

#### 3.1 Numerically stable and online softmax

`softmax(x)_i = exp(x_i) / sum_j exp(x_j)` overflows for scores as small as 89
in fp32; the standard fix subtracts the row max `m` first (it cancels in the
ratio), leaving two per-row statistics: the max `m` and the normalizer
`l = sum_j exp(x_j - m)`. The blocker for tiling is that `m` and `l` are
row-global — they depend on all `N` scores, but a tile sees only `Bc` of them.
The online-softmax insight (Milakov & Gimelshein 2018) is that `(m, l)` and the
partial output `O` can be updated incrementally. If the running statistics are
`(m_prev, l_prev, O_prev)` and the current tile yields `(m_tile, l_tile,
PV_tile)`, the merge is:

```text
m_new = max(m_prev, m_tile)
l_new = exp(m_prev - m_new) * l_prev + exp(m_tile - m_new) * l_tile
O_new = ( l_prev * exp(m_prev - m_new) * O_prev
          + exp(m_tile - m_new) * PV_tile ) / l_new
```

The correction factors re-express previously computed exponentials relative to
the new max. After the last tile, `O` equals exact attention — no approximation.

#### 3.2 Lessons from FlashAttention v1 to apply

* **Tiling for SRAM.** Choose `Br`, `Bc` so the `Q`, `K`, `V` tiles and the
  `Br x Bc` score tile all fit in shared memory. **State your shared-memory
  budget and the resulting occupancy** (Part 4a asks you to derive it).
* **Online softmax fused into the loop.** Carry `m`, `l`, and the output
  accumulator across `K/V` tiles; rescale the accumulator when `m` updates.
* **One kernel, no HBM round-trips** for the intermediate scores.

The scaffold provides the tiling loops, the SRAM declarations, and the `Q/K/V`
loads. The three `TODO` blocks are:

* `ASSIGN1_3_1`: tile scores (`QK^T * scale` into shared `S`) and the tile row
  max.
* `ASSIGN1_3_2`: the tile softmax numerator `exp(S - row_m)` (use `__expf`) and
  the tile row sum.
* `ASSIGN1_3_3`: the online merge of Section 3.1 into `O`, `m`, `l`.

**Deliverables.**
* Fused kernel passing the Part 1 harness, causal and non-causal
  (`kernel_tests/grade_flash.py`).
* A **plot of HBM traffic vs. the Part 2 baseline** across `N in {512, 1024,
  2048, 4096}`.
* A table of your chosen `Br`, `Bc` and the resulting shared-memory usage per
  block.

### Part 4: Adapting the kernel to context length and block size

A single set of tile sizes rarely performs well across all sequence lengths.
Here you analyze how your kernel's design should change as a function of context
length `N`, block size `Br x Bc`, and head dimension `d`, reasoning explicitly
about the shared-memory budget and occupancy trade-offs that govern the choice.
Everything in Part 4 goes in your report.

#### 4a. Shared-memory budget

Derive the shared-memory footprint of one thread block in your fused kernel as a
function of `Br`, `Bc`, and `d` for fp32 inputs, accounting for the `Q`, `K`,
and `V` tiles and the `Br x Bc` score tile. Then:

* Identify which term grows as the **product** `Br * Bc` rather than the sum,
  and explain why it dominates the budget as tiles grow.
* For your GPU, state the shared-memory-per-SM limit and compute the largest
  `Br = Bc` you can use at `d = 64` while keeping **at least two resident blocks
  per SM**. Show your arithmetic.

#### 4b. Regime analysis

For each context-length regime, state (i) whether the kernel is closer to
compute-bound or memory-bound, (ii) the primary risk to performance, and (iii)
how you would adjust `Br`, `Bc`, or the parallelization strategy. Justify each
in 2-3 sentences.

1. **Short (`N <= 1024`).** Consider the case where `batch x heads` is too small
   to fill all SMs.
2. **Medium (`N = 2K-8K`).**
3. **Long (`N >= 16K`).** Address both the accumulated cost of the
   online-softmax bookkeeping and numerical stability.

#### 4c. Empirical sweep

Using `bench/sweep.py`, benchmark your fused kernel across `Br, Bc in {8, 16,
32}` for `N in {64, 128, 512, 1024, 4096}` at `d = 64`. Produce:

* A table or heatmap of achieved **TFLOPs/s** for each `(N, Br, Bc)`.
* A paragraph explaining the observed tile-size behavior at the short and long
  ends of the sweep. State whether the best `(Br, Bc)` changes. If it does not,
  explain why the same configuration wins on this GPU and identify the relevant
  parallelism, shared-memory, register, or loop-overhead constraint.

### How to test

All commands run from `attn_kernel_assignment/`. **After every kernel edit,
recompile** with
`bash compile_cuda.sh` before running the `ctypes`-based graded tests.

* **Reference/harness (Part 1):** `python kernel_tests/test_reference.py`
* **Baseline (Part 2):** `python kernel_tests/grade_naive.py`
* **Fused kernel (Part 3), non-causal:** `python kernel_tests/grade_flash.py --causal 0`
* **Fused kernel (Part 3), causal:** `python kernel_tests/grade_flash.py --causal 1`
* **Sweep (Part 4c):** `python bench/sweep.py`

Kernel tests compare against your Part 1 reference at several shapes and exit
nonzero on any failure. Debugging tips: `printf` works inside CUDA kernels
(guard with `if (bx==0 && by==0 && tx==0)`); if `N = 32` (a single tile) passes
but larger `N` fails, your online merge is not order-independent — recheck the
correction factors; if causal fails only for `N > 32`, you are comparing
tile-local instead of global positions.

### Submission

You must submit:

* Your three completed source files, with every `ASSIGN` block filled in:
  `reference/attention_ref.py`, `src/naive_attention_kernel.cu`, and
  `src/flash_attention_kernel.cu`.
* A report with 1) a detailed account of your approach, and 2) your answers and
  explanations to all questions, in text and screenshots. The report must
  contain:
  * your group name, and the name and EID of every member;
  * the output of all graded tests;
  * **Part 2:** the baseline runtime + HBM-bytes table, and the `N` at which you
    run out of memory;
  * **Part 3:** the HBM-traffic-vs-baseline plot and the tile-size /
    shared-memory table;
  * **Q1 (Part 4a):** the shared-memory-budget derivation, the identification of
    the `Br * Bc` term, and the largest-`Br=Bc` arithmetic for your GPU;
  * **Q2 (Part 4b):** the three-regime analysis;
  * **Q3 (Part 4c):** the TFLOPs/s sweep grid and the explanatory paragraph.
    Where your empirical optimum disagrees with your Part 4a prediction, explain
    the discrepancy (e.g. register pressure, tensor-core tile alignment, or
    launch overhead).

Collect the three files on the node, from `attn_kernel_assignment/`:

```bash
tar czf submission.tar.gz reference/attention_ref.py \
    src/naive_attention_kernel.cu src/flash_attention_kernel.cu
```

Copy it to **your own machine** — the node's disk is destroyed on `stop`, and
Canvas cannot reach the node. `gpulease.py status` prints the key and host;
from a terminal on your machine:

```bash
scp -i ~/.ssh/gpulease_<group> \
    ubuntu@<host>:~/<your-repo>/attn_kernel_assignment/submission.tar.gz .
```

Check the file actually arrived **before** you run `gpulease.py stop`.

**Naming format:** name the PDF report `assignment1_report.pdf`, then bundle it
with the archive you copied off the node — no need to unpack anything:

```bash
tar czf assignment1_{Group Name}.tar.gz submission.tar.gz assignment1_report.pdf
```

Submission is to be done through [Canvas](https://utexas.instructure.com/courses/1451468){:target="_blank"}. Only one person per group is
required to submit.

### Grading

* Part 1 — reference and harness (`test_reference.py`): **10%**
* Part 2 — baseline kernel (`grade_naive.py`): **20%**
* Part 3 — fused Flash Attention kernel: **40%**
  * Forward correctness, non-causal (`grade_flash.py --causal 0`): 25%
  * Causal masking (`grade_flash.py --causal 1`): 10%
  * HBM-traffic plot and tile-size table: 5%
* Part 4 — analysis report: **30%**
  * Q1 (4a, shared-memory budget): 10%
  * Q2 (4b, regime analysis): 10%
  * Q3 (4c, empirical sweep + discrepancy explanation): 10%

Kernel scores are all-or-nothing per test.

### FAQ / Common issues

* **`cannot open shared object file`** — you skipped `bash compile_cuda.sh`, or
  you are on a freshly started node. Run it from `attn_kernel_assignment/`.
* **`head_dim too large for shared memory`** — the graded shapes keep
  `d <= 64`; this usually means a broken index writing out of bounds, not a real
  limit.
* **Everything fails after one bad run** — an illegal memory access poisons the
  CUDA context. Exit the Python process, then rerun.
* **Correct for `N = 32`, wrong for larger `N`** — your online merge is
  order-dependent; recheck the `exp(m_prev - m_new)` correction factors.
* **Causal wrong only in the last rows/columns** — off-by-one in the
  global-position translation; the diagonal (`k == q`) is *kept*.

### Acknowledgments

This assignment builds on
[Assignment 4 of 11-868 *Large Language Model Systems*](https://llmsystem.github.io/llmsystemhomework/assignment_4/){:target="_blank"}
at Carnegie Mellon University, on
[flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal){:target="_blank"},
and on materials from a previous offering of this course. We thank those authors
for making their materials publicly available.

