from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host import make_zero_state, make_RX, make_RY, apply_single_qubit_gate, apply_cnot
from qucu_sim import QuCu
import numpy as np
import time


n_qubits_array=[2,5,8,10,12,15,18]
n_layer_array=[1,2,3,4,5,6,8,10]
num_runs=5
CSV_PATH = ROOT / "cuda_ablation_results.csv"

def run_gpu(n_qubits, n_layers, n_runs=num_runs):
    sim = QuCu(n_qubits)
    kernel_times = []; wall_times = []; probs = None
    for run in range(n_runs + 5):
        sim.reset(); sim.kernel_time_ms = 0.0
        wall_start = time.perf_counter()
        for _ in range(n_layers):
            for q in range(n_qubits): # all qubits
                sim.rx(target=q, theta=np.pi/2)
            for q in range(n_qubits):# all qubits
                sim.ry(target=q, theta=np.pi/3)
            sim.cnot(control=0, target=1)
        wall_ms = (time.perf_counter() - wall_start) * 1000
        if run < 5: continue  #  warmup
        kernel_times.append(sim.kernel_time_ms)
        wall_times.append(wall_ms)
        probs = list(sim.ExpVal_Z())
    return np.mean(kernel_times), np.mean(wall_times), probs

def run_cpu(n_qubits, n_layers):
    state = make_zero_state(n_qubits)
    Rx    = make_RX(np.pi/2)
    Ry    = make_RY(np.pi/3)
    start = time.perf_counter()
    for _ in range(n_layers):
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, n_qubits, target=q, U=Rx)
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, n_qubits, target=q, U=Ry)
        state = apply_cnot(state, n_qubits, control=0, target=1)
    wall_ms = (time.perf_counter() - start) * 1000
    probs   = [abs(state[i])**2 for i in range(2**n_qubits)]
    return wall_ms, probs

def compare(n_qubits, n_layers):
    cpu_wall, cpu_probs        = run_cpu(n_qubits, n_layers)
    gpu_k, gpu_wall, gpu_probs = run_gpu(n_qubits, n_layers)
    errors  = [abs(cpu_probs[i] - gpu_probs[i]) for i in range(len(cpu_probs))]
    mean_err = np.mean(errors)
    max_err  = np.max(errors)
    status   = "PASS" if max_err < 1e-6 else "FAIL"
    return cpu_wall, gpu_k, gpu_wall, mean_err, max_err, status

csv_rows = []

#experiment 1: varying qubit count, fixed 6 layers
print("Experiment 1: Varying qubit count (fixed 6 layers)")
print(f"{'Qubits':<8} {'CPU Wall (ms)':<16} {'GPU Kernel (ms)':<18} {'GPU Wall (ms)':<16} {'Mean Error':<14} {'Max Error':<12} {'Status'}")

for n in n_qubits_array:
    cpu_wall, gpu_k, gpu_wall, mean_err, max_err, status = compare(n, 6)
    print(f"{n:<8} {cpu_wall:<16.6f} {gpu_k:<18.6f} {gpu_wall:<16.6f} {mean_err:<14.2e} {max_err:<12.2e} {status}")
    csv_rows.append(["vary_qubits", n, 6, cpu_wall, gpu_k, gpu_wall, mean_err, max_err, status])

# experiment 2: varying depth, fixed 15 qubits 
print("\nExperiment 2: Varying circuit depth (fixed 15 qubits)")
print(f"{'Layers':<8} {'CPU Wall (ms)':<16} {'GPU Kernel (ms)':<18} {'GPU Wall (ms)':<16} {'Mean Error':<14} {'Max Error':<12} {'Status'}")

for l in n_layer_array:
    cpu_wall, gpu_k, gpu_wall, mean_err, max_err, status = compare(15, l)
    print(f"{l:<8} {cpu_wall:<16.6f} {gpu_k:<18.6f} {gpu_wall:<16.6f} {mean_err:<14.2e} {max_err:<12.2e} {status}")
    csv_rows.append(["vary_depth", 15, l, cpu_wall, gpu_k, gpu_wall, mean_err, max_err, status])

# write CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Qubits", "Layers", "CPU_Wall_ms", "GPU_Kernel_ms", "GPU_Wall_ms", "Mean_Error", "Max_Error", "Status"])
    writer.writerows(csv_rows)

print(f"\nResults saved to {CSV_PATH}")
