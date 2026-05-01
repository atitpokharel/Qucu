import numpy as np
import time

# ── gate definitions ──────────────────────────────────────────
def make_zero_state(n_qubits):
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state

def make_RX(theta):
    c = np.cos(theta / 2); s = np.sin(theta / 2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def make_RY(theta):
    c = np.cos(theta / 2); s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def make_RZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

# ── gate applications ─────────────────────────────────────────
def apply_single_qubit_gate(state, n_qubits, target, U):
    dim = 2**n_qubits; step = 2**(n_qubits - 1 - target); period = step * 2
    for base in range(0, dim, period):
        for i in range(step):
            i0 = base + i; i1 = i0 + step
            a0, a1 = state[i0], state[i1]
            state[i0] = U[0,0]*a0 + U[0,1]*a1
            state[i1] = U[1,0]*a0 + U[1,1]*a1
    return state

def apply_cnot(state, n_qubits, control, target):
    dim          = 2**n_qubits
    control_mask = 2**(n_qubits - 1 - control)
    target_mask  = 2**(n_qubits - 1 - target)
    for idx in range(dim):
        if (idx & control_mask) != 0:
            flipped = idx ^ target_mask
            if idx < flipped:
                state[idx], state[flipped] = state[flipped], state[idx]
    return state

# ── vqc forward pass ──────────────────────────────────────────
def run_vqc_cpu(n_qubits, n_layers, theta_rx=np.pi/2, theta_ry=np.pi/3):
    state = make_zero_state(n_qubits)
    Rx    = make_RX(theta_rx)
    Ry    = make_RY(theta_ry)

    start = time.perf_counter()
    for _ in range(n_layers):
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, n_qubits, target=q, U=Rx)
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, n_qubits, target=q, U=Ry)
        state = apply_cnot(state, n_qubits, control=0, target=1)
    wall_ms = (time.perf_counter() - start) * 1000

    probs = [abs(state[i])**2 for i in range(2**n_qubits)]
    return wall_ms, probs

# ── benchmark ─────────────────────────────────────────────────
if __name__ == "__main__":

    # experiment 1: varying qubit count, fixed 6 layers
    print("Experiment 1: Varying qubit count (fixed 6 layers)")
    print(f"{'Qubits':<8} {'Wall (ms)':<15}")
    print("-" * 25)
    for n in [2, 5, 8, 10, 12, 15, 18]:
        wall_ms, probs = run_vqc_cpu(n, n_layers=6)
        print(f"{n:<8} {wall_ms:<15.6f}")

    # experiment 2: varying depth, fixed 15 qubits
    print("\nExperiment 2: Varying circuit depth (fixed 15 qubits)")
    print(f"{'Layers':<8} {'Wall (ms)':<15}")
    print("-" * 25)
    for l in [1, 2, 3, 4, 5, 6, 8, 10]:
        wall_ms, probs = run_vqc_cpu(15, n_layers=l)
        print(f"{l:<8} {wall_ms:<15.6f}")