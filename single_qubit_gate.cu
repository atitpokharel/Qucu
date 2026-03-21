#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/single_qubit_gate.cuh"

int main() {
    // test case from checkpoint 1: Ry(pi/3) on |1>,  expected psi[0]=-0.5, psi[1]=0.8660
    int n_qubits = 1;
    int dim      = 1 << n_qubits; //2^n_qubits
    size_t size  = dim * sizeof(cuDoubleComplex);

    cuDoubleComplex U00, U01, U10, U11;
    make_RY(M_PI / 3.0, &U00, &U01, &U10, &U11); //swap for make_RX or make_RZ as needed

    cuDoubleComplex* h_state = (cuDoubleComplex*)malloc(size);
    h_state[0] = make_cuDoubleComplex(0.0, 0.0); //initialize
    h_state[1] = make_cuDoubleComplex(1.0, 0.0);

    cuDoubleComplex* d_state; //device state vector
    cudaMalloc((void**)&d_state, size);                         //allocate on GPU
    cudaMemcpy(d_state, h_state, size, cudaMemcpyHostToDevice); //host to device

    int block_size = 256;
    int grid_size  = ((1 << (n_qubits - 1)) + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    //launch the kernel
    single_qubit_gate_kernel<<<grid_size, block_size>>>(
        d_state, n_qubits, 0, U00, U01, U10, U11);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_state, d_state, size, cudaMemcpyDeviceToHost); //device -> host

    printf("GPU: psi[0]=%.4f  psi[1]=%.4f\n", cuCreal(h_state[0]), cuCreal(h_state[1]));
    printf("Kernel time: %.6f ms\n", ms);

    //verification against expected values
    double expected0 = -0.5, expected1 = 0.8660254037;
    double err0 = fabs(cuCreal(h_state[0]) - expected0);
    double err1 = fabs(cuCreal(h_state[1]) - expected1);
    printf("Max error: %.2e  %s\n", fmax(err0, err1), fmax(err0, err1) < 1e-10 ? "Done" : "Error");

    free(h_state);  //free host mem
    cudaFree(d_state); //free gpu mem
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}