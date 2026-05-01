from ctypes import CDLL, POINTER, c_double, c_int, c_void_p
from pathlib import Path

import numpy as np


_LIB = None


def _load_library():
    global _LIB
    if _LIB is not None:
        return _LIB

    root = Path(__file__).resolve().parents[1]
    lib_path = root / "qucu.so"
    lib = CDLL(str(lib_path))

    lib.qucu_create.argtypes = [c_int]
    lib.qucu_create.restype = c_void_p

    lib.qucu_destroy.argtypes = [c_void_p]
    lib.qucu_destroy.restype = None

    lib.qucu_reset.argtypes = [c_void_p]
    lib.qucu_reset.restype = None

    lib.qucu_rx.argtypes = [c_void_p, c_int, c_double]
    lib.qucu_rx.restype = c_double

    lib.qucu_ry.argtypes = [c_void_p, c_int, c_double]
    lib.qucu_ry.restype = c_double

    lib.qucu_rz.argtypes = [c_void_p, c_int, c_double]
    lib.qucu_rz.restype = c_double

    lib.qucu_cnot.argtypes = [c_void_p, c_int, c_int]
    lib.qucu_cnot.restype = c_double

    lib.qucu_state_dim.argtypes = [c_void_p]
    lib.qucu_state_dim.restype = c_int

    lib.qucu_expval_z.argtypes = [c_void_p, POINTER(c_double), c_int]
    lib.qucu_expval_z.restype = None

    lib.qucu_get_kernel_time_ms.argtypes = [c_void_p]
    lib.qucu_get_kernel_time_ms.restype = c_double

    _LIB = lib
    return _LIB


class QuCuSim:
    def __init__(self, n_qubits):
        self._lib = _load_library()
        self._ptr = self._lib.qucu_create(n_qubits)
        if not self._ptr:
            raise RuntimeError("Failed to create QuCuSim backend")

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._lib.qucu_destroy(ptr)
            self._ptr = None

    def reset(self):
        self._lib.qucu_reset(self._ptr)

    def rx(self, target, theta):
        return self._lib.qucu_rx(self._ptr, target, theta)

    def ry(self, target, theta):
        return self._lib.qucu_ry(self._ptr, target, theta)

    def rz(self, target, theta):
        return self._lib.qucu_rz(self._ptr, target, theta)

    def cnot(self, control, target):
        return self._lib.qucu_cnot(self._ptr, control, target)

    def ExpVal_Z(self):
        dim = self._lib.qucu_state_dim(self._ptr)
        probs = np.empty(dim, dtype=np.float64)
        self._lib.qucu_expval_z(
            self._ptr,
            probs.ctypes.data_as(POINTER(c_double)),
            dim,
        )
        return probs

    @property
    def kernel_time_ms(self):
        return self._lib.qucu_get_kernel_time_ms(self._ptr)
