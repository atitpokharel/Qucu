CC = nvcc
FLAGS = -O2 -std=c++14 -Xcompiler -fPIC -shared -arch=sm_80

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

.PHONY: all clean setup_env compile run plot

all: setup_env compile

setup_env:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
		$(PIP) install --upgrade pip; \
		$(PIP) install numpy matplotlib; \
	else \
		echo "Virtual environment already exists."; \
	fi

compile: setup_env
	$(CC) $(FLAGS) \
		-Iutils \
		bindings/qucu_bindings.cu \
		-o qucu.so \
		-lcudart

run: compile
	$(PYTHON) Python_sim/circuit_ablation.py

plot: setup_env
	$(PYTHON) Python_sim/plot_ablation.py

clean:
	rm -f qucu.so
	rm -rf $(VENV_DIR)
