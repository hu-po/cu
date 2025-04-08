# cu

CUDA playground

verify cuda version

```bash
nvidia-smi
nvcc --version
```

compile examples

```bash
mkdir build
cd build
cmake ..
make
```

run examples

```bash
./bin/allreduce
```

