# An-Adaptive-Multi-Interval-Numerical-Method-with-Neural-Guided-Mesh-Refinement
A parametric deep learning–assisted numerical framework that integrates a Parameter-Geometry Adaptive Deep Neural Network (PGA-DNN) with a high-fidelity Adaptive Implicit–Explicit Local Differential Transform Method coupled with Multi-Interval Chebyshev Spectral Collocation (AIELDTM-MIChSCM) solver, designed for the efficient and robust solution of nonlinear one-dimensional advection–diffusion–reaction (ADR) equations, particularly in strongly advection-dominated regimes characterized by thin boundary layers.

# Code Availability Notice
The complete source code associated with this work — including all benchmark experiments, the full AIELDTM-MIChSCM solver, and the PGA-DNN training pipeline — will be released after the corresponding manuscript has been accepted for publication.
Only a partial implementation is available at this stage. Prior to publication, this repository contains only the baseline IELDTM solver for the one-dimensional Burgers–Fisher equation, located in the burgers_fisher_codes folder. This portion is shared to provide reviewers and interested readers with a working reference for the underlying numerical method.
The full implementation — including:

> the multi-interval extension IELDTM-MIChSCM,

> its adaptive variant AIELDTM-MIChSCM,

> the PGA-DNN model 

For early access requests (e.g., for peer-review purposes), please contact the corresponding author.

# Requirements
PyTorch
NumPy, SciPy, SymPy
Matplotlib
Pandas, openpyxl

