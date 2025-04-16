# MNIST Digit Recognition in C using Leptonica

This project implements a handwritten digit recognition neural network in C using the Leptonica image processing library.

## Features
- Neural network with manual matrix operations
- Forward and backward propagation
- ReLU and softmax activation functions
- Leptonica used for image loading and processing
- Demonstrate network with example images(0-9)

## Prerequisites
- Leptonica (install via Homebrew: `brew install leptonica`)
- MNIST dataset (use raw IDX files)
- Make sure MNIST files (train-images.idx3-ubyte, train-labels.idx1-ubyte) are placed in the correct directory.

## Build Instructions

Compile all `.c` files using:

```bash
gcc *.c -o mnist `pkg-config --cflags --libs lept`

- ./mnist to run