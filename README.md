# QuCu (Kuku)

A CUDA-accelerated quantum state vector simulator for variational quantum circuits (VQC).  
QuCu offloads single-qubit gate and CNOT gate operations to the GPU using custom CUDA kernels, and exposes them to Python via `ctypes`.

## Authors

Atit Pokharel, Ratun Rahman  
Department of Electrical and Computer Engineering, UAH

**Source code / Dev version:** [https://github.com/atitpokharel/Qucu](https://github.com/atitpokharel/Qucu)  
*Note: The repository contains the dev version of this project tailored for a general NVIDIA GPU platform. The version described in this README is specifically configured and tailored for the ASAX cluster system.*

---

## Dependencies

| Dependency       | Version / Notes                                        |
|------------------|--------------------------------------------------------|
| **CUDA Toolkit** | 11.0+ (module `cuda` on ASA/RCHAU)                    |
| **nvcc**         | Provided by the CUDA module                            |
| **Python**       | 3.9+ (system Python on ASA/RCHAU)                      |
| **numpy**        | Installed automatically via `make`                     |
| **matplotlib**   | Installed automatically via `make` (for plotting only) |

No external Python C headers (`Python.h`) are needed. The binding uses `ctypes` to load the compiled `qucu.so` shared library.

---

## Repository Structure

```
Qucu/
├── Makefile                    # Build automation (venv, compile, run, plot)
├── submission.pbs              # PBS job script for ASA/RCHAU
├── run_me.sh                   # One-command submit + tail output
├── setup.py                    # Minimal package metadata
│
├── utils/
│   ├── single_qubit_gate.cuh   # RX, RY, RZ CUDA kernel + gate matrix helpers
│   └── cnot_gate.cuh           # CNOT CUDA kernel
│
├── bindings/
│   ├── qucu_bindings.cu        # C-API wrapper (extern "C") around QuCuSim struct
│   └── qucu_ctypes.py          # Python ctypes loader for qucu.so
│
├── Python_sim/
│   ├── qucu_sim.py             # High-level Python QuCu class
│   ├── host.py                 # CPU-only reference implementation
│   ├── circuit_ablation.py     # CPU vs GPU benchmark (writes CSV + prints table)
│   ├── plot_ablation.py        # Generates plots from CSV results
│   └── vqc.py                  # Simple VQC driver script
│
├── Nsys/                       # Nsight Systems profiling scripts
│   ├── qucu_sim.py             # Profiler wrapper
│   ├── bench.py                # Single-qubit gate benchmark
│   └── bench_cnot.py           # CNOT gate benchmark
```

---

## Build & Run on ASA/RCHAU

### Step 1: Upload and extract

```bash
# EXTRACT THE TAR FILE ON ASA
tar -xzf Qucu.tar.gz
cd Qucu
```

### Step 2: Submit the job (one command)

```bash
chmod +x run_me.sh
./run_me.sh
# and wait for the job to finish.
```

This will:
1. Remove any stale output file (`qucu_simulation.qsub_out`)
2. Submit `submission.pbs` to the `classgpu` queue via `qsub`
3. Wait for the output file to appear
4. Tail the output with `less +F` (press `Ctrl+C` then `q` to exit)

###  PBS job

The `submission.pbs` script requests 1 Ampere GPU node and runs:

```
module load cuda
make clean       # remove old .venv and qucu.so
make all         # create venv, install deps, compile qucu.so
make run         # run circuit_ablation.py (prints table + writes CSV)
make plot        # generate plots from CSV into plots/
```

### Step 3: Check results

After the job completes:

```bash
# View the terminal output
cat qucu_simulation.qsub_out

# View the CSV results
cat cuda_ablation_results.csv

# View generated plots
ls plots/
# exp1_vary_qubits.png
# exp2_vary_depth.png
```

---

## Running Locally (with a CUDA GPU)

```bash
cd Qucu

# Make sure nvcc is available
nvcc --version

# Build and run
make clean && make all && make run && make plot
```

---

## Makefile Targets

| Target      | Description                                              |
|-------------|----------------------------------------------------------|
| `make all`  | Create venv (if needed) + compile `qucu.so`              |
| `make run`  | Compile + run `circuit_ablation.py`                      |
| `make plot` | Generate plots from `cuda_ablation_results.csv`          |
| `make clean`| Remove `qucu.so` and `.venv`                             |

---

## Input Data

No external input data is required. The benchmark circuit is generated programmatically in `circuit_ablation.py`:

- **Circuit**: For each layer, apply RX(π/2) to all qubits, then RY(π/3) to all qubits, then CNOT(0, 1).
- **Correctness**: GPU results are compared against the CPU reference implementation (`host.py`). A test passes if the max probability error is < 1e-6.

---

## Key Parameters

Parameters are configured at the top of `circuit_ablation.py`:

| Parameter         | Default                        | Description                                |
|-------------------|--------------------------------|--------------------------------------------|
| `n_qubits_array`  | `[2, 5, 8, 10, 12, 15, 18]`   | Qubit counts for Experiment 1              |
| `n_layer_array`   | `[1, 2, 3, 4, 5, 6, 8, 10]`   | Layer depths for Experiment 2              |
| `num_runs`        | `5`                            | Timed runs per config (after 5 warmups)    |

PBS resource parameters are in `submission.pbs`:

| Parameter    | Default                  | Description                        |
|--------------|--------------------------|------------------------------------|
| Queue        | `classgpu`               | GPU queue on ASA/RCHAU             |
| GPU          | `ampere`, 1 GPU          | GPU type and count                 |
| Memory       | `8000mb`                 | Requested RAM                      |
| Walltime     | `01:00:00`               | Max job duration                   |

---

## Notes

- `ExpVal_Z()` returns computational-basis probabilities from the state vector (single D2H transfer).
- The `ctypes` binding (`bindings/qucu_ctypes.py`) loads `qucu.so` from the project root.
- GPU architecture is set to `sm_80` (Ampere) in the Makefile. Change `-arch=sm_80` for other GPU architectures.
