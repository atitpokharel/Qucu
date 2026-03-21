# QuCu (Kuku)
A CUDA-accelerated quantum state vector simulator for variational quantum circuits.

## Structure
```
QuCu/
  utils/
    single_qubit_gate.cuh   # single-qubit gate kernel
    cnot_gate.cuh            # CNOT kernel
  bindings/
    qucu_bindings.cu         # pybind11 bindings
  Python_implementation/
    host.py                  # CPU reference implementation
  Python_sim/
    qucu_sim.py              # Python wrapper class
    circuit_ablation.py      # benchmarking script
  single_qubit_gate.cu       # single-qubit gate test
  cnot_gate.cu               # CNOT test
  vqc.cu                     # VQC forward pass
  setup.py                   # build script
```

## Build
```bash
# compile shared library
nvcc -O2 -std=c++14 \
     -Xcompiler -fPIC \
     -shared \
     -I$(python -c "import pybind11; print(pybind11.get_include())") \
     -I/usr/local/cuda/include \
     -Iutils \
     -I$(python -c "import sysconfig; print(sysconfig.get_path('include'))") \
     bindings/qucu_bindings.cu \
     -o qucu.so \
     -lcudart
```

## Usage
```python
from Python_sim.qucu_sim import QuCu
import numpy as np

sim = QuCu(n_qubits=2)
sim.rx(target=0, theta=np.pi/2)
sim.ry(target=1, theta=np.pi/3)
sim.cnot(control=0, target=1)
probs = sim.measure()
print(probs)
```

## Run Benchmarks
```bash
python Python_sim/circuit_ablation.py
```

## Requirements
- CUDA Toolkit
- pybind11
- numpy

## Authors
Atit Pokharel, Ratun Rahman
Department of Electrical and Computer Engineering, UAH