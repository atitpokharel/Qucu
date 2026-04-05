import sys
sys.path.append("/home/ap1284@DS.UAH.edu/Atit/GPU_Computing/Qucu")

from qucu_sim import QuCu
import numpy as np
import time

def run_single_gate_isolated(n_qubits, n_runs=5):
    sim = QuCu(n_qubits)
    kernel_times = []

    for run in range(n_runs + 5):
        sim.reset()
        sim.kernel_time_ms = 0.0

        wall_start = time.perf_counter()
        sim.rx(target=0, theta=np.pi/2)  # single isolated gate call
        wall_ms = (time.perf_counter() - wall_start) * 1000

        if run < 5:
            continue  # warmup
        kernel_times.append(sim.kernel_time_ms)

    return np.mean(kernel_times)

print("Isolated single_qubit_gate_kernel benchmark (Rx on qubit 0)")
print(f"{'Qubits':<10} {'Kernel Time (ms)':<20} {'State Vector Size (MB)'}")

for n in [18]:
    size_mb = (2**n * 16) / (1024**2)  #cuDoubleComplex=16b
    k_time  = run_single_gate_isolated(n)
    print(f"{n:<10} {k_time:<20.6f} {size_mb:.2f}")