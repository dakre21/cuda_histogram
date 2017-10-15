#!/bin/bash
nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello
