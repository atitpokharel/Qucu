import sys
sys.path.append("/home/ap1284@DS.UAH.edu/Atit/GPU_Computing/Qucu")

import qucu
import numpy as np

class QuCu:
    def __init__(self, n_qubits):
        self.n_qubits       = n_qubits
        self.state          = qucu.zeros(n_qubits)
        self.kernel_time_ms = 0.0

    def reset(self):
        self.state          = qucu.zeros(self.n_qubits) #reset to |00...0>
        self.kernel_time_ms = 0.0

    def rx(self, target, theta):
        self.kernel_time_ms += qucu.rx(self.state, self.n_qubits, target, theta)

    def ry(self, target, theta):
        self.kernel_time_ms += qucu.ry(self.state, self.n_qubits, target, theta)

    def rz(self, target, theta):
        self.kernel_time_ms += qucu.rz(self.state, self.n_qubits, target, theta)

    def cnot(self, control, target):
        self.kernel_time_ms += qucu.cnot(self.state, self.n_qubits, control, target)

    def measure(self):
        return qucu.measure(self.state)