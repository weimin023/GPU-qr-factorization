#!/usr/bin/env bash

export PYTHONPATH="/workspace/.cutedsl:/workspace/.cutedsl/nvidia_cutlass_dsl/python_packages:${PYTHONPATH}"

# NVIDIA_VISIBLE_DEVICES=all is valid for container runtimes, but
# CUDA_VISIBLE_DEVICES=all makes PyTorch/CUDA see zero devices.
if [ "${CUDA_VISIBLE_DEVICES:-}" = "all" ]; then
  unset CUDA_VISIBLE_DEVICES
fi
