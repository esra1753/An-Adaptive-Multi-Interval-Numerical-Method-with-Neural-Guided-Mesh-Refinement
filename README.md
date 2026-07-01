# An-Adaptive-Multi-Interval-Numerical-Method-with-Neural-Guided-Mesh-Refinement
This repository contains selected Python implementations associated with the study:

An Adaptive Multi-Interval Numerical Method with Neural-Guided Mesh Refinement for Parametric ADR Equations

The proposed framework combines a Parameter-Geometry Adaptive Deep Neural Network (PGA-DNN) with a high-fidelity Adaptive Implicit–Explicit Local Differential Transform Method coupled with Multi-Interval Chebyshev Spectral Collocation (AIELDTM-MIChSCM) solver. The method is designed for the efficient and robust numerical solution of nonlinear one-dimensional advection–diffusion–reaction (ADR) equations, especially in strongly advection-dominated regimes characterized by thin boundary layers.

## Overview
The repository focuses on one-dimensional nonlinear ADR-type problems, with particular emphasis on Burgers–Fisher-type benchmark equations.

The main computational components are:

* IELDTM-ChSCM: the baseline Implicit–Explicit Local Differential Transform Method coupled with Chebyshev spectral collocation in time.
* IELDTM-MIChSCM: the multi-interval extension of the baseline solver.
* AIELDTM-MIChSCM: the adaptive multi-interval solver using non-uniform spatial meshes.
* PGA-DNN / Data-driven MLP pipeline: a neural-network-assisted training framework for learning parametric solution behavior and guiding adaptive mesh refinement.

## Mathematical Model

The main benchmark problem considered in the codes is the one-dimensional Burgers–Fisher-type equation:

$u_t + \alpha u u_x = \varepsilon u_{xx} + \beta u(1-u),
\qquad x \in [0,1], \quad t \in [0,t_f].$

The exact solution is as follows 

$u(x,t) = \frac{1}{2} + \frac{1}{2} \tanh(\delta_1 x - \delta_1 \delta_2 t)$

where $\(\delta_1 = -\alpha/4\varepsilon\)$ and $\(\delta_2 = \alpha/2 + 2 \varepsilon \beta / \alpha\).$ The analytical solution is used to define the initial condition, boundary conditions, and error evaluation.

## Available Code Files
### 1. Baseline IELDTM Solver

The baseline solver implements the IELDTM-ChSCM formulation for the one-dimensional Burgers–Fisher equation.

Main features:

* Solves the one-dimensional Burgers–Fisher equation.
* Uses Chebyshev–Gauss–Lobatto collocation nodes in time.
* Constructs the Chebyshev time-differentiation matrix.
* Applies the local differential transform recurrence in space.
* Enforces $C^0$ and $C^1$ continuity between adjacent spatial intervals.
* Compares the numerical solution with the analytical solution.
* Computes the maximum absolute error.

This file provides the fundamental numerical structure on which the multi-interval and adaptive solvers are based.

### 2. Multi-Interval Solver with Adaptive Mesh Support

This code extends the baseline IELDTM-ChSCM formulation to a multi-interval time discretization. The time interval $[0,t_f]$ is divided into several subintervals, and each subinterval can use its own spatial mesh.

Main features:

* Solves the problem over multiple time subintervals.
* Allows different spatial meshes in different time intervals.
* Supports both uniform and adaptive mesh configurations.
* Includes an optional Excel-based adaptive mesh input.
* Uses quintic Hermite projection to transfer Taylor coefficients between consecutive meshes.
* Preserves boundary compatibility when transferring data from one time subinterval to the next.
* Computes numerical and analytical solutions.
* Produces the essential final visual outputs.

The final visualization includes:

* Exact solution
* Approximate solution
* Error heatmap

#### Adaptive Mesh Input

Adaptive meshes can be loaded from an Excel file. The Excel file should contain one adaptive mesh per column. Each column corresponds to one time subinterval:

$[t_k,t_{k+1}], \qquad k=0,1,\ldots,K-1.$

Each column must contain strictly increasing spatial points in the interval $[0,1]$, including both endpoints $0$ and $1$.

Example usage:
```python
x_mesh_list = load_adaptive_meshes_from_excel(
    "adaptive_mesh.xlsx",
    expected_K=len(t_points)-1
)
```
If this line is commented out, the solver uses the default uniform mesh.

### 3. PGA-DNN / Data-Driven Adaptive Training Code

This script implements the neural-network-assisted component of the framework. A data-driven multilayer perceptron is trained to approximate the parametric solution map:

$(x,t,\varepsilon) \mapsto u(x,t;\varepsilon).$

Main features:

* Generates numerical solution data using the IELDTM-MIChSCM solver.
* Trains a data-driven neural network on the generated solution data.
* Uses $(x,t,\varepsilon)$ as the neural network input.
* Applies adaptive training cycles.
* Selects challenging diffusion parameters using a PINN-residual indicator.
* Constructs residual-guided adaptive non-uniform spatial meshes.
* Augments the training set with adaptive-mesh solution data produced by the AIELDTM-MIChSCM.
* Evaluates the trained model against the analytical solution.
* Produces a compact final heatmap visualization.

The final visualization contains:

* Exact solution heatmap
* Approximate neural-network solution heatmap
* Absolute error heatmap

## Requirements

The codes require the following Python packages:

- numpy
- sympy
- scipy
- matplotlib
- torch
- openpyxl
