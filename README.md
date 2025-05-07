# MNIST Digit Recognition in C using Leptonica

This project was developed by me and my friends from IIT Jodhpur for the course *CSL1010*. It implements a handwritten digit recognition neural network in C, using the Leptonica image processing library.

## Features
- Neural network built from scratch using manual matrix operations
- Forward and backward propagation
- ReLU and softmax activation functions
- Leptonica used for image loading and preprocessing
- Demonstration using example digit images (0–9)

## Prerequisites
- Leptonica (install via Homebrew: `brew install leptonica`)
- MNIST dataset (use the raw IDX files)

Make sure the following MNIST files are placed in the project directory:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Build Instructions

To compile all `.c` files and run the project, use the following commands:

```bash
gcc *.c -o mnist `pkg-config --cflags --libs lept`
./mnist
