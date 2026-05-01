#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void cnot_kernel(cuDoubleComplex* state, int n_qubits, int control, int target){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //one thread per state vector index
    int dim = 1 << n_qubits;
    if (idx >= dim) return;

    int control_mask = 1 << (n_qubits - 1 - control); //mask for control bit
    int target_mask  = 1 << (n_qubits - 1 - target); //mask for target bit

    if ((idx & control_mask) != 0){
        int flipped = idx ^ target_mask; //flip the target bit
        if (idx < flipped){ //process each pair once
            cuDoubleComplex tmp = state[idx];
            state[idx] = state[flipped];
            state[flipped] = tmp;}} //swap amplitudes
}
