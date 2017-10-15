#!/bin/bash
nvcc histogram.cu -L /usr/local/cuda/lib -lcudart -o histogram
