from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT     = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "cuda_ablation_results.csv"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

exp1 = [r for r in rows if r["Experiment"] == "vary_qubits"]
exp2 = [r for r in rows if r["Experiment"] == "vary_depth"]

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CPU_COLOR    = "#d62728"
GPU_K_COLOR  = "#1f77b4"
GPU_W_COLOR  = "#2ca02c"

def plot_experiment(x_vals, x_label, cpu_wall, gpu_kernel, gpu_wall, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    ax1.semilogy(x_vals, cpu_wall,   "o-", color=CPU_COLOR,   linewidth=2, markersize=7, label="CPU Wall")
    ax1.semilogy(x_vals, gpu_kernel, "s-", color=GPU_K_COLOR, linewidth=2, markersize=7, label="GPU Kernel")
    ax1.semilogy(x_vals, gpu_wall,   "^-", color=GPU_W_COLOR, linewidth=2, markersize=7, label="GPU Wall")

    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("Time (ms, log scale)", fontsize=12)
    ax1.set_title("Execution Time", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    ax1.set_xticks(x_vals)

    speedup_kernel = np.array(cpu_wall) / np.array(gpu_kernel)
    speedup_wall   = np.array(cpu_wall) / np.array(gpu_wall)

    ax2.plot(x_vals, speedup_kernel, "s-", color=GPU_K_COLOR, linewidth=2, markersize=7, label="Speedup (kernel)")
    ax2.plot(x_vals, speedup_wall,   "^-", color=GPU_W_COLOR, linewidth=2, markersize=7, label="Speedup (wall)")

    peak_idx = np.argmax(speedup_kernel)
    ax2.annotate(f"{speedup_kernel[peak_idx]:.0f}×",
                 xy=(x_vals[peak_idx], speedup_kernel[peak_idx]),
                 xytext=(0, 12), textcoords="offset points",
                 ha="center", fontsize=10, color=GPU_K_COLOR, fontweight="bold")

    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel("Speedup (CPU / GPU)", fontsize=12)
    ax2.set_title("GPU Speedup over CPU", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    ax2.set_xticks(x_vals)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


qubits     = [int(r["Qubits"])        for r in exp1]
cpu_w1     = [float(r["CPU_Wall_ms"]) for r in exp1]
gpu_k1     = [float(r["GPU_Kernel_ms"]) for r in exp1]
gpu_w1     = [float(r["GPU_Wall_ms"]) for r in exp1]

plot_experiment(qubits, "Qubits", cpu_w1, gpu_k1, gpu_w1,
                "Experiment 1: Varying Qubit Count (6 Layers)",
                "exp1_vary_qubits.png")

layers     = [int(r["Layers"])        for r in exp2]
cpu_w2     = [float(r["CPU_Wall_ms"]) for r in exp2]
gpu_k2     = [float(r["GPU_Kernel_ms"]) for r in exp2]
gpu_w2     = [float(r["GPU_Wall_ms"]) for r in exp2]

plot_experiment(layers, "Layers", cpu_w2, gpu_k2, gpu_w2,
                "Experiment 2: Varying Circuit Depth (15 Qubits)",
                "exp2_vary_depth.png")

print(f"\nAll plots saved to {PLOT_DIR}/")
