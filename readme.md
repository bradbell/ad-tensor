# An AD Overloading of the Libtorch Tensor Class

## Release
This is not a true release, but rather a preliminay version
meant for testing.

## Purpose
-   Provide simple clear C++ code 
    for the following operator overloading techniques
    -   Recording a function of both parameters and variables.
    -   Forward mode for function values and derivatives w.r.t variables.
    -   Revere mode for derivatives w.r.t. variables.
    -   Optimization of recorded operations.
    -   Sparsity calculations.
    -   AD Checkpointing with arbitrary order derivatives.
    -   User defined atomic function with arbitrary order derivatives.

## Benchmarks
The benchmarks seem to indicate that some of the ad-tensor options
are faster than Libtorch's autograd.
You can run them on your own machine; see
https://ad-tensor.readthedocs.io/en/latest/cmake.html#test

