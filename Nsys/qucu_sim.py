import sys
sys.path.append("/home/ap1284@DS.UAH.edu/Atit/GPU_Computing/Qucu")

import qucu
import numpy as np

class QuCu:
    def __init__(self, n_qubits):
        self.n_qubits       = n_qubits
        self._sim           = qucu.QuCuSim(n_qubits) # persistent device state
        self.kernel_time_ms = 0.0

    def reset(self):
        self._sim.reset()# single H2D transfer
        self.kernel_time_ms = 0.0
        self._sim.kernel_time_ms = 0.0

    def rx(self, target, theta):
        self.kernel_time_ms += self._sim.rx(target, theta)

    def ry(self, target, theta):
        self.kernel_time_ms += self._sim.ry(target, theta)

    def rz(self, target, theta):
        self.kernel_time_ms += self._sim.rz(target, theta)

    def cnot(self, control, target):
        self.kernel_time_ms += self._sim.cnot(control, target)

    def ExpVal_Z(self):
        return self._sim.ExpVal_Z()# single D2H transfer
