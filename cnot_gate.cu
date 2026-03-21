#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils/cnot_gate.cuh"

int main(){
    // test case from checkpoint 1: CNOT on Ry(pi/3)|1> x |1>
    // expected: psi[01]=-0.5, psi[10]=0.8660
    int n_qubits = 2;
    int dim      = 1 << n_qubits; //2^n_qubits
    size_t size  = dim * sizeof(cuDoubleComplex);

    // state after Ry(pi/3) on qubit 0 applied to |11>
    cuDoubleComplex* h_state = (cuDoubleComplex*)malloc(size);
    h_state[0] = make_cuDoubleComplex( 0.0,    0.0); //|00>
    h_state[1] = make_cuDoubleComplex(-0.5,    0.0); //|01>
    h_state[2] = make_cuDoubleComplex( 0.0,    0.0); //|10>
    h_state[3] = make_cuDoubleComplex( 0.8660, 0.0); //|11>

    cuDoubleComplex* d_state;
    cudaMalloc((void**)&d_state, size); //allocate on GPU
    cudaMemcpy(d_state, h_state, size, cudaMemcpyHostToDevice); //host to device

    int block_size = 256;
    int grid_size  = (dim + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    //launch the kernel
    cnot_kernel<<<grid_size, block_size>>>(d_state, n_qubits, 0, 1); //launch the kernel

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_state, d_state, size, cudaMemcpyDeviceToHost); //device to host

    printf("GPU: psi[00]=%.4f  psi[01]=%.4f  psi[10]=%.4f  psi[11]=%.4f\n",
           cuCreal(h_state[0]), cuCreal(h_state[1]),
           cuCreal(h_state[2]), cuCreal(h_state[3]));
    printf("Kernel time: %.6f ms\n", ms);

    free(h_state);//free host mem
    cudaFree(d_state);//free gpu mem
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}