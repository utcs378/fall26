---
layout: page
permalink: /assignments/assignment0
title: "Assignment 0: PyTorch & Profiling"
---
#### **Released:** TODO <br/> **Due:** TODO
{: .no_toc}

* (The list will be replaced with the table of contents.)
{:toc}

### Part 0: Overview and Setup
#### Overview
This is a warm-up assignment to get you comfortable with **PyTorch**, the framework we will use all semester, and with **profiling**, the skill of measuring what your code actually does on the hardware.

The assignment has two parts:
1. **PyTorch Fundamentals.** You will train a small MLP that classifies handwritten digits (MNIST). Instead of calling PyTorch's built-in layers, you will implement the core operators by hand, including both their **forward** and **backward** (gradient) passes. You will then assemble them into an MLP, write the training loop, and tune it to reach high accuracy.
2. **The PyTorch Profiler.** You will profile your training loop with `torch.profiler` to see the underlying CPU and GPU activity, read the resulting traces and memory timelines, and reason about the results from a systems perspective.

All of your work happens in the provided notebook, `Assignment0_PyTorch_Profiling.ipynb`.

#### Setup
This assignment is designed to run on **Google Colab**, which gives you free access to a GPU.
1. Open the [Assignment 0 notebook](https://colab.research.google.com/drive/1UceXiCg9ypi4NbxjKV1TbZo5qZy_OL30?usp=sharing){:target="\_blank"} in Colab and make your own copy.
2. Go to **Runtime → Change runtime type**, set **Hardware accelerator** to **GPU** (a free **T4** is plenty), and click **Save**.
3. Click **Connect** in the top-right corner.
4. Run the **Setup** cell. It should print `Using device: cuda`. If it prints `cpu`, revisit the steps above.

`torch` and `torchvision` are pre-installed on Colab, so there is nothing to install. The notebook downloads the MNIST dataset automatically the first time you run it.

**Note**: The code will still run on CPU, but training and profiling are much slower, and the GPU-only parts of Part 2 (the CUDA memory timeline) will be skipped. We strongly recommend using a GPU runtime.

### Part 1: PyTorch Fundamentals
Complete the code blocks between `# TODO: ...` and `# END OF YOUR CODE` in the notebook. You will implement, in order:
* The Linear operator
* The ReLU operator
* The Cross-Entropy loss
* The Multilayer Perceptron (MLP) model
* The training loop

Each operator ships with a **check cell** that verifies your implementation against PyTorch's reference using `torch.autograd.gradcheck` and `torch.allclose`. Make sure every check prints its ✔ before moving on.

#### **Question 1**: Model size
Compute the number of trainable parameters in the MLP when `hidden_dim=64`, and how much memory (in KB) is needed to store the parameter tensors at 32-bit floating-point precision. Show each step of your calculation.

#### **Question 2**: Training recipe
Find a combination of training parameters (`hidden_dim`, `batch_size`, `optim_method`, `lr`, `epochs`) that reaches a **test accuracy > 97%**. Report your recipe and attach the loss curve.

### Part 2: Profiling Your Code
Complete the code blocks between `# TODO: ...` and `# END OF YOUR CODE` in the notebook.
Run the provided `run_profile(n_step, hidden_dim, batch_size, optim_method)` function to profile a few training steps. It records CPU and GPU activity, labels the phases of each step (`h2d`, `forward`, `clear grad`, `backward`, `optimizer`) with `record_function`, and exports:
* a **trace** (`*_trace.json`) you can open in [Perfetto](https://ui.perfetto.dev), and
* a **CUDA memory timeline** (`*_memory.html`, GPU only).

#### **Question 3**: Batch size and GPU bubbles
Generate traces at two batch sizes with `run_profile(n_step=5, hidden_dim=256, batch_size=..., optim_method="adam")`, then compare `batch_size=32` vs. `batch_size=256`. Attach a screenshot of each GPU lane and explain which has more idle gaps (bubbles), and why.

#### **Question 4**: Optimizer memory footprint
Compare the memory timelines of an SGD run and an Adam run. Explain the differences and the relationship between the sizes of the parameters, gradients, and optimizer states (including the effect of the optional SGD `momentum` buffer).

#### **Question 5**: Step time vs. batch size
Measure the time per training step across several batch sizes and reason about how it scales — is it linear in the batch size? Explain in terms of GPU utilization and fixed per-step overhead.

### Submission
<!-- TODO: confirm the exact submission mechanics for the course (Gradescope / Canvas / Ed). -->
You must submit:
* Your completed notebook `Assignment0_PyTorch_Profiling.ipynb`, with **all cells executed**, outputs visible (check cells passing, training logs, loss curves), and all questions answered with text and screenshots when required.

**Naming Format**:
* Please name the notebook file as `assignment0_{Your EID}.ipynb`.

### Grading
<!-- TODO: confirm the point breakdown with the course staff. The split below is a placeholder. -->
* Part 1
    * Implementations: TODO%
    * Question 1: TODO%
    * Question 2: TODO%
* Part 2
    * Implementations: TODO%
    * Question 3: TODO%
    * Question 4: TODO%
    * Question 5: TODO%
