#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/single_qubit_gate.cuh"
#include "utils/cnot_gate.cuh"

int main(){
    // vqc forward pass: 2 qubits, 2 layers (Rx, Ry, CNOT)
    // matches checkpoint 1 python vqc test case exactly
    int n_qubits = 2;
    int n_layers = 2;
    int dim = 1 << n_qubits; //2^n_qubits
    size_t size  = dim * sizeof(cuDoubleComplex);

    // initialize |00> state
    cuDoubleComplex* h_state = (cuDoubleComplex*)malloc(size);
    h_state[0] = make_cuDoubleComplex(1.0, 0.0); //|00>
    h_state[1] = make_cuDoubleComplex(0.0, 0.0);
    h_state[2] = make_cuDoubleComplex(0.0, 0.0);
    h_state[3] = make_cuDoubleComplex(0.0, 0.0);

    cuDoubleComplex* d_state;//device state vector
    cudaMalloc((void**)&d_state, size); //allocate on GPU
    cudaMemcpy(d_state, h_state, size, cudaMemcpyHostToDevice); //host to device

    int block_size = 256;
    int sq_pairs = 1 << (n_qubits - 1); //2^(n_qubits-1) pairs
    int sq_grid = (sq_pairs + block_size - 1) / block_size; //grid for single qubit gate
    int cnot_grid = (dim + block_size - 1) / block_size; //grid for cnot

    cuDoubleComplex U00, U01, U10, U11;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int layer = 0; layer < n_layers; layer++){
        // Rx(pi/2) on qubit 0
        make_RX(M_PI/2.0, &U00, &U01, &U10, &U11);
        single_qubit_gate_kernel<<<sq_grid, block_size>>>(d_state, n_qubits, 0, U00, U01, U10, U11);

        // Rx(pi/2) on qubit 1
        single_qubit_gate_kernel<<<sq_grid, block_size>>>(d_state, n_qubits, 1, U00, U01, U10, U11);

        // Ry(pi/3) on qubit 0
        make_RY(M_PI/3.0, &U00, &U01, &U10, &U11);
        single_qubit_gate_kernel<<<sq_grid, block_size>>>(d_state, n_qubits, 0, U00, U01, U10, U11);

        // Ry(pi/3) on qubit 1
        single_qubit_gate_kernel<<<sq_grid, block_size>>>(d_state, n_qubits, 1, U00, U01, U10, U11);

        // CNOT: control=0, target=1
        cnot_kernel<<<cnot_grid, block_size>>>(d_state, n_qubits, 0, 1);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_state, d_state, size, cudaMemcpyDeviceToHost); //device to host

    printf("Final state:\n");
    printf("  psi[00]=%.4f  psi[01]=%.4f\n", cuCreal(h_state[0]), cuCreal(h_state[1]));
    printf("  psi[10]=%.4f  psi[11]=%.4f\n", cuCreal(h_state[2]), cuCreal(h_state[3]));

    // probabilities
    printf("Measurement probabilities:\n");
    for (int i = 0; i < dim; i++)
        printf("  p[%d] = %.4f\n", i, cuCreal(h_state[i])*cuCreal(h_state[i]) +
                                       cuCimag(h_state[i])*cuCimag(h_state[i]));
    printf("Kernel time: %.6f ms\n", ms);

    free(h_state); //free host mem
    cudaFree(d_state); //free gpu mem
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}