#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include "single_qubit_gate.cuh"
#include "cnot_gate.cuh"

namespace py = pybind11;

cuDoubleComplex* to_device(py::array_t<std::complex<double>> state){
    auto buf    = state.request();
    size_t size = buf.size * sizeof(cuDoubleComplex);
    cuDoubleComplex* d_state;
    cudaMalloc((void**)&d_state, size); //allocate on GPU
    cudaMemcpy(d_state, buf.ptr, size, cudaMemcpyHostToDevice); //host to device
    return d_state;
}

void to_host(cuDoubleComplex* d_state, py::array_t<std::complex<double>> state){
    auto buf    = state.request();
    size_t size = buf.size * sizeof(cuDoubleComplex);
    cudaMemcpy(buf.ptr, d_state, size, cudaMemcpyDeviceToHost); //device to host
    cudaFree(d_state);
}

// helper: create events, record, synchronize, return elapsed time in ms
float get_kernel_time(cudaEvent_t start, cudaEvent_t stop){
    float ms = 0.0f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

float apply_rx(py::array_t<std::complex<double>> state, int n_qubits, int target, double theta){
    cuDoubleComplex U00, U01, U10, U11;
    make_RX(theta, &U00, &U01, &U10, &U11);

    cuDoubleComplex* d_state = to_device(state);
    int block_size = 256;
    int grid_size  = ((1 << (n_qubits - 1)) + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    single_qubit_gate_kernel<<<grid_size, block_size>>>(d_state, n_qubits, target, U00, U01, U10, U11);

    cudaEventRecord(stop);
    float ms = get_kernel_time(start, stop);
    to_host(d_state, state);
    return ms; //kernel time in ms
}

float apply_ry(py::array_t<std::complex<double>> state, int n_qubits, int target, double theta){
    cuDoubleComplex U00, U01, U10, U11;
    make_RY(theta, &U00, &U01, &U10, &U11);

    cuDoubleComplex* d_state = to_device(state);
    int block_size = 256;
    int grid_size  = ((1 << (n_qubits - 1)) + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    single_qubit_gate_kernel<<<grid_size, block_size>>>(d_state, n_qubits, target, U00, U01, U10, U11);

    cudaEventRecord(stop);
    float ms = get_kernel_time(start, stop);
    to_host(d_state, state);
    return ms; //kernel time in ms
}

float apply_rz(py::array_t<std::complex<double>> state, int n_qubits, int target, double theta){
    cuDoubleComplex U00, U01, U10, U11;
    make_RZ(theta, &U00, &U01, &U10, &U11);

    cuDoubleComplex* d_state = to_device(state);
    int block_size = 256;
    int grid_size  = ((1 << (n_qubits - 1)) + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    single_qubit_gate_kernel<<<grid_size, block_size>>>(d_state, n_qubits, target, U00, U01, U10, U11);

    cudaEventRecord(stop);
    float ms = get_kernel_time(start, stop);
    to_host(d_state, state);
    return ms; //kernel time in ms
}

float apply_cnot(py::array_t<std::complex<double>> state, int n_qubits, int control, int target){
    cuDoubleComplex* d_state = to_device(state);
    int block_size = 256;
    int grid_size  = ((1 << n_qubits) + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    cnot_kernel<<<grid_size, block_size>>>(d_state, n_qubits, control, target);

    cudaEventRecord(stop);
    float ms = get_kernel_time(start, stop);
    to_host(d_state, state);
    return ms; //kernel time in ms
}

py::array_t<double> measure(py::array_t<std::complex<double>> state){
    auto buf   = state.request();
    int  dim   = buf.size;
    auto probs = py::array_t<double>(dim);
    auto pbuf  = probs.request();
    auto* ptr  = (std::complex<double>*)buf.ptr;
    auto* pptr = (double*)pbuf.ptr;
    for (int i = 0; i < dim; i++)
        pptr[i] = std::norm(ptr[i]); //|psi|^2
    return probs;
}

PYBIND11_MODULE(qucu, m){
    m.doc() = "QuCu: CUDA-accelerated quantum state vector simulator";
    m.def("zeros", [](int n_qubits){
        int dim = 1 << n_qubits;
        auto state = py::array_t<std::complex<double>>(dim);
        auto buf   = state.request();
        auto* ptr  = (std::complex<double>*)buf.ptr;
        std::fill(ptr, ptr + dim, std::complex<double>(0.0, 0.0));
        ptr[0] = std::complex<double>(1.0, 0.0); //|00...0>
        return state;
    }, "initialize state to |00...0>");
    m.def("rx",    &apply_rx,   "apply Rx(theta), returns kernel time in ms");
    m.def("ry",    &apply_ry,   "apply Ry(theta), returns kernel time in ms");
    m.def("rz",    &apply_rz,   "apply Rz(theta), returns kernel time in ms");
    m.def("cnot",  &apply_cnot, "apply CNOT, returns kernel time in ms");
    m.def("measure", &measure,  "return measurement probabilities");
}
