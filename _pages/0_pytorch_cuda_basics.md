---
layout: page
permalink: /assignments/assignment0
title: "Assignment 0: PyTorch & Profiling"
---
#### **Released:** 08/25/2026 <br/> **Due:** 08/30/2026
{: .no_toc}

* (The list will be replaced with the table of contents.)
{:toc}

### Part 0: Overview and Setup
#### Overview
This is a warm-up assignment to get you comfortable with **PyTorch**, the framework we will use all semester, and with **profiling**, the skill of measuring what your code actually does on the hardware.

The assignment has two parts:
1. **PyTorch Fundamentals.** You will train a small MLP that classifies handwritten digits (MNIST). Instead of calling PyTorch's built-in layers, you will implement the core operators by hand, including both their **forward** and **backward** (gradient) passes. You will then assemble them into an MLP, write the training loop, and tune it to reach high accuracy.
2. **The PyTorch Profiler.** You will profile your training loop with `torch.profiler` to see the underlying CPU and GPU activity, read the resulting traces and memory timelines, and reason about the results from a systems perspective.

Make sure you complete the code blocks between `# TODO: ...` and `# END OF YOUR CODE` before handing in the assignment.

In addition to the notebook, you need to submit a separate report answering all the questions 
.

You can find the notebook with detailed instructions here: <a href="{{ '/assets/assignments/Assignment0_PyTorch_Profiling.ipynb' | relative_url }}" download>**Assignment0_PyTorch_Profiling.ipynb**</a> (click to download).

#### Setup
This assignment is designed to run on **Google Colab**, which gives you free access to a GPU.
1. Download the Assignment 0 notebook, then open it in Colab (**File → Upload notebook**) and make your own copy.
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
How many trainable parameters are there in this MLP model when hidden_dim=64? By default PyTorch uses 32-bit floating point number for the parameters. How much memory (in KB) do we need to store the parameter tensors? Show each step of your calculation process.
> Hint: You can print `sum(p.numel() for p in model.parameters())` to check if your calculation of parameter number is correct

#### **Question 2**: Training recipe
Find a combination of training parameters (`hidden_dim`, `batch_size`, `optim_method`, `lr`, `epochs`) that reaches a **test accuracy > 97%**. Report your recipe and attach the loss curve produced by `plot_histories` for that run.

### Part 2: Profiling Your Code
Complete the code blocks between `# TODO: ...` and `# END OF YOUR CODE` in the notebook.
Run the provided `run_profile(hidden_dim, batch_size, optim_method, n_step)` function to profile a few training steps. It records CPU and GPU activity, labels the phases of each step (`h2d`, `forward`, `clear grad`, `backward`, `optimizer`) with `record_function`, and exports:
* a **trace** (`*_trace.json`) you can open in [Perfetto](https://ui.perfetto.dev), and
* a **CUDA memory timeline** (`*_memory.html`).

#### **Question 3**: Batch size and GPU kernels
First generate the two traces by running the profiler at two batch sizes:

```python
run_profile(hidden_dim=256, batch_size=32,  optim_method="adam")   # -> adam_h=256_bs=32_trace.json
run_profile(hidden_dim=256, batch_size=1024, optim_method="adam")   # -> adam_h=256_bs=1024_trace.json
```

Open `adam_h=256_bs=32_trace.json` and `adam_h=256_bs=1024_trace.json` in Perfetto, then zoom in to **the last step** (you should see a tag like `ProfilerStep#4`).
* Attach a screenshot of the **GPU track** during **the last step** from each trace. What are the names of the GPU kernels executed during forward and backward passes of the Linear operator? Compare the execution time of GPU kernels from the two traces.

#### **Question 4**: Optimizer memory footprint
Generate the memory timelines for an SGD run and an Adam run (this requires a GPU runtime):

```python
run_profile(hidden_dim=256, batch_size=128, optim_method="sgd")    # -> sgd_h=256_bs=128_memory.html
run_profile(hidden_dim=256, batch_size=128, optim_method="adam")   # -> adam_h=256_bs=128_memory.html
```

* **Q4.1** Open each `*_memory.html` file in a browser. What are the differences in memory consumption between the two optimizers? Explain the relationship between the sizes of the **parameters**, the **gradients**, and the **optimizer states**. Attach corresponding screenshots of your memory timelines.

* **Q4.2** PyTorch's SGD optimizer has an optional `momentum` arg (see the [documentation](https://docs.pytorch.org/docs/2.13/generated/torch.optim.SGD.html)), which adds a momentum term to the optimization algorithm when set to a non-zero number. Overwrite the `make_optimizer` function in the code block below to add a new optimization method, `sgd_momentum`, that uses SGD optimizer with `momentum=0.9` in `make_optimizer`. Run the profiling code with the new optimization method, and compare the memory timeline. What changed, and why? Attach corresponding screenshots of your memory timelines.


#### **Question 5**: Step time vs. batch size
Measure how the **time of one training step** changes with the batch size. Run the profiler at several batch sizes, for example:

```python
for bs in [32, 128, 512, 2048]:
    run_profile(hidden_dim=256, batch_size=bs, optim_method="adam")
```

* **Q5.1** For each batch size, open the trace in Perfetto and measure the wall-clock duration of the **last training step**. Report the step time for each batch size and attach a screenshot of your measurement.
    > Hint: Two ways to get the execution time of a certain operation or a tagged period from the trace: 1) hover over the block, or 2) drag to select an arbitrary period and read the measured time from the timeline

* **Q5.2** How does the step time change with the batch size? Going from `bs=32` to `bs=2048` increases the input workload by 64×; does the step time increase by anywhere near 64×? Reason about the step time change with what you observe in terms of GPU utilization, CPU activities, and data movements.

### Submission
You must submit:
* Your completed notebook `Assignment0_PyTorch_Profiling.ipynb`, with **all cells executed** and outputs visible (check cells passing, training logs, loss curves, etc.).
* A report with your answers and explainations to all questions in text and screenshots.

**Naming Format**:
* Please name the notebook file as `assignment0_{Your EID}.ipynb` and the PDF report as `assignment0_report_{Your EID}.pdf` when submitting.

### Grading
In this assignment we will only grade the report.
* Question 1: 20%
* Question 2: 20%
* Question 3: 20%
* Question 4: 20%
    * Question 4.1: 10%
    * Question 4.2: 10%
* Question 5: 20%
    * Question 5.1: 10%
    * Question 5.2: 10%
