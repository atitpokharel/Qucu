#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

inline __host__ __device__ cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) 
{
    return cuCadd(a, b);
}

inline __host__ __device__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) 
{
    return cuCmul(a, b);
}

__global__ void single_qubit_gate_kernel(cuDoubleComplex* state, int n_qubits, int target,
                                         cuDoubleComplex U00, cuDoubleComplex U01,
                                         cuDoubleComplex U10, cuDoubleComplex U11){
    int tid       = blockIdx.x * blockDim.x + threadIdx.x; //one thread per amplitude pair
    int num_pairs = 1 << (n_qubits - 1); //2^(n_qubits-1) amplitude pairs (bit shift method)
    if (tid >= num_pairs) return;

    int step = 1 << (n_qubits - 1 - target); //step size for the target bit
    int low  = tid & (step - 1); //bits below target (bitwise AND)
    int high = tid >> (n_qubits - 1 - target);//bits above target (RIGHT SHIFT)
    int i0   = high * 2 * step + low; //target bit = 0
    int i1   = i0 + step; //target bit = 1

    cuDoubleComplex a0 = state[i0]; //amplitude for target bit = 0
    cuDoubleComplex a1 = state[i1]; //amplitude for target bit = 1

    state[i0] = U00 * a0 + U01 * a1; //update pair
    state[i1] = U10 * a0 + U11 * a1;
}

// gate matrix helpers
void make_RX(double theta, cuDoubleComplex* U00, cuDoubleComplex* U01,
                           cuDoubleComplex* U10, cuDoubleComplex* U11){
    double c = cos(theta / 2.0), s = sin(theta / 2.0);
    *U00 = make_cuDoubleComplex( c,  0.0);
    *U01 = make_cuDoubleComplex( 0.0, -s);
    *U10 = make_cuDoubleComplex( 0.0, -s);
    *U11 = make_cuDoubleComplex( c,  0.0);}

void make_RY(double theta, cuDoubleComplex* U00, cuDoubleComplex* U01,
                           cuDoubleComplex* U10, cuDoubleComplex* U11){
    double c = cos(theta / 2.0), s = sin(theta / 2.0);
    *U00 = make_cuDoubleComplex( c, 0.0);
    *U01 = make_cuDoubleComplex(-s, 0.0);
    *U10 = make_cuDoubleComplex( s, 0.0);
    *U11 = make_cuDoubleComplex( c, 0.0);}

void make_RZ(double theta, cuDoubleComplex* U00, cuDoubleComplex* U01,
                           cuDoubleComplex* U10, cuDoubleComplex* U11){
    *U00 = make_cuDoubleComplex(cos(theta/2.0), -sin(theta/2.0));
    *U01 = make_cuDoubleComplex(0.0, 0.0);
    *U10 = make_cuDoubleComplex(0.0, 0.0);
    *U11 = make_cuDoubleComplex(cos(theta/2.0),  sin(theta/2.0));}
