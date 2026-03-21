from setuptools import setup, Extension
import pybind11
import subprocess
import os

def get_cuda_include():
    return "/usr/local/cuda/include" 

def get_cuda_lib():
    return "/usr/local/cuda/lib64"

ext = Extension(
    name="qucu",
    sources=["bindings/qucu_bindings.cpp"],
    include_dirs=[
        pybind11.get_include(),
        get_cuda_include(),
        "utils"  #so bindings can find .cuh files
    ],
    library_dirs=[get_cuda_lib()],
    libraries=["cudart"], #link against cuda runtime
    extra_compile_args=["-O2", "-std=c++14"],
    language="c++"
)

setup(
    name="qucu",
    ext_modules=[ext]
)