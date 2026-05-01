#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include "single_qubit_gate.cuh"
#include "cnot_gate.cuh"

namespace py = pybind11;

struct QuCuSim {
    cuDoubleComplex* d_state; // persistent device state vector
    int n_qubits;
    int dim;
    size_t size;
    float kernel_time_ms;

    QuCuSim(int n_qubits) : n_qubits(n_qubits) {
        dim  = 1 << n_qubits;
        size = dim * sizeof(cuDoubleComplex);
        cudaMalloc((void**)&d_state, size); // allocate once
        reset();
    }

    ~QuCuSim() {
        cudaFree(d_state); // free once
    }

    void reset() {
        // initialize |00...0> on host and transfer once
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
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        cnot_kernel<<<grid_size, block_size>>>(d_state, n_qubits, control, target);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        kernel_time_ms += ms;
        return ms;
    }

    py::array_t<double> ExpVal_Z() {
        // single D2H transfer only at measurement
        std::vector<cuDoubleComplex> h_state(dim);
        cudaMemcpy(h_state.data(), d_state, size, cudaMemcpyDeviceToHost);

        auto probs = py::array_t<double>(dim);
        auto pbuf  = probs.request();
        auto* pptr = (double*)pbuf.ptr;
        for (int i = 0; i < dim; i++)
            pptr[i] = cuCreal(h_state[i]) * cuCreal(h_state[i])
                    + cuCimag(h_state[i]) * cuCimag(h_state[i]);
        return probs;
    }

private:
    float launch_single_qubit(int target,
                               cuDoubleComplex U00, cuDoubleComplex U01,
                               cuDoubleComplex U10, cuDoubleComplex U11) {
        int block_size = 256;
        int grid_size  = ((1 << (n_qubits - 1)) + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        single_qubit_gate_kernel<<<grid_size, block_size>>>(
            d_state, n_qubits, target, U00, U01, U10, U11);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        kernel_time_ms += ms;
        return ms;
    }
};

PYBIND11_MODULE(qucu, m) {
    m.doc() = "QuCu: CUDA-accelerated quantum state vector simulator";
    py::class_<QuCuSim>(m, "QuCuSim")
        .def(py::init<int>())
        .def("reset",    &QuCuSim::reset)
        .def("rx",       &QuCuSim::apply_rx)
        .def("ry",       &QuCuSim::apply_ry)
        .def("rz",       &QuCuSim::apply_rz)
        .def("cnot",     &QuCuSim::apply_cnot)
        .def("ExpVal_Z", &QuCuSim::ExpVal_Z)
        .def_readwrite("kernel_time_ms", &QuCuSim::kernel_time_ms);
}
