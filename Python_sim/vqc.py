from qucu_sim import QuCu
import numpy as np
import time

if __name__ == "__main__":
    n_qubits =10
    n_layers = 5
    sim = QuCu(n_qubits)

    for run in range(3):
        sim.reset()
        sim.kernel_time_ms = 0.0

        wall_start = time.perf_counter()
        for _ in range(n_layers):
            sim.rx(target=0, theta=np.pi/2)
            sim.rx(target=1, theta=np.pi/2)
            sim.ry(target=0, theta=np.pi/3)
            sim.ry(target=1, theta=np.pi/3)
            sim.cnot(control=0, target=1)
        wall_ms = (time.perf_counter() - wall_start) * 1000

        print(f"Run {run+1}: kernel={sim.kernel_time_ms:.6f} ms  wall={wall_ms:.4f} ms  overhead={wall_ms - sim.kernel_time_ms:.4f} ms")