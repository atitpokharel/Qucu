#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <new>
#include <vector>
#include "single_qubit_gate.cuh"
#include "cnot_gate.cuh"

struct QuCuSim {
    cuDoubleComplex* d_state;
    int n_qubits;
    int dim;
    size_t size;
    float kernel_time_ms;

    QuCuSim(int n_qubits) : d_state(nullptr), n_qubits(n_qubits), kernel_time_ms(0.0f) {
        dim  = 1 << n_qubits;
        size = dim * sizeof(cuDoubleComplex);
        cudaMalloc((void**)&d_state, size);
        reset();
    }

    ~QuCuSim() {
        if (d_state != nullptr) {
            cudaFree(d_state);
        }
    }

    void reset() {
        std::vector<cuDoubleComplex> h_state(dim, make_cuDoubleComplex(0.0, 0.0));
        h_state[0] = make_cuDoubleComplex(1.0, 0.0);
        cudaMemcpy(d_state, h_state.data(), size, cudaMemcpyHostToDevice);
        kernel_time_ms = 0.0f;
    }

    float apply_rx(int target, double theta) {
        cuDoubleComplex U00, U01, U10, U11;
        make_RX(theta, &U00, &U01, &U10, &U11);
        return launch_single_qubit(target, U00, U01, U10, U11);
    }

    float apply_ry(int target, double theta) {
        cuDoubleComplex U00, U01, U10, U11;
        make_RY(theta, &U00, &U01, &U10, &U11);
        return launch_single_qubit(target, U00, U01, U10, U11);
    }

    float apply_rz(int target, double theta) {
        cuDoubleComplex U00, U01, U10, U11;
        make_RZ(theta, &U00, &U01, &U10, &U11);
        return launch_single_qubit(target, U00, U01, U10, U11);
    }

    float apply_cnot(int control, int target) {
        int block_size = 256;
        int grid_size  = (dim + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cnot_kernel<<<grid_size, block_size>>>(d_state, n_qubits, control, target);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        kernel_time_ms += ms;
        return ms;
    }

    void expval_z(double* out_probs, int out_len) {
        if (out_probs == nullptr || out_len < dim) {
            return;
        }

        std::vector<cuDoubleComplex> h_state(dim);
        cudaMemcpy(h_state.data(), d_state, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < dim; i++) {
            out_probs[i] = cuCreal(h_state[i]) * cuCreal(h_state[i])
                         + cuCimag(h_state[i]) * cuCimag(h_state[i]);
        }
    }

private:
    float launch_single_qubit(int target,
                              cuDoubleComplex U00, cuDoubleComplex U01,
                              cuDoubleComplex U10, cuDoubleComplex U11) {
        int block_size = 256;
        int grid_size  = ((1 << (n_qubits - 1)) + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        single_qubit_gate_kernel<<<grid_size, block_size>>>(
            d_state, n_qubits, target, U00, U01, U10, U11);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        kernel_time_ms += ms;
        return ms;
    }
};

extern "C" {

void* qucu_create(int n_qubits) {
    return new (std::nothrow) QuCuSim(n_qubits);
}

void qucu_destroy(void* sim_ptr) {
    delete static_cast<QuCuSim*>(sim_ptr);
}

void qucu_reset(void* sim_ptr) {
    if (sim_ptr != nullptr) {
        static_cast<QuCuSim*>(sim_ptr)->reset();
    }
}

double qucu_rx(void* sim_ptr, int target, double theta) {
    return sim_ptr != nullptr
        ? static_cast<double>(static_cast<QuCuSim*>(sim_ptr)->apply_rx(target, theta))
        : 0.0;
}

double qucu_ry(void* sim_ptr, int target, double theta) {
    return sim_ptr != nullptr
        ? static_cast<double>(static_cast<QuCuSim*>(sim_ptr)->apply_ry(target, theta))
        : 0.0;
}

double qucu_rz(void* sim_ptr, int target, double theta) {
    return sim_ptr != nullptr
        ? static_cast<double>(static_cast<QuCuSim*>(sim_ptr)->apply_rz(target, theta))
        : 0.0;
}

double qucu_cnot(void* sim_ptr, int control, int target) {
    return sim_ptr != nullptr
        ? static_cast<double>(static_cast<QuCuSim*>(sim_ptr)->apply_cnot(control, target))
        : 0.0;
}

int qucu_state_dim(void* sim_ptr) {
    return sim_ptr != nullptr ? static_cast<QuCuSim*>(sim_ptr)->dim : 0;
}

void qucu_expval_z(void* sim_ptr, double* out_probs, int out_len) {
    if (sim_ptr != nullptr) {
        static_cast<QuCuSim*>(sim_ptr)->expval_z(out_probs, out_len);
    }
}

double qucu_get_kernel_time_ms(void* sim_ptr) {
    return sim_ptr != nullptr
        ? static_cast<double>(static_cast<QuCuSim*>(sim_ptr)->kernel_time_ms)
        : 0.0;
}

}
