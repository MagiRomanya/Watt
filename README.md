# Watt

**Watt** is a collection of simple, easy-to-use, **header-only libraries** designed to provide common computations for **physics-based simulation** in **Computer Graphics**. Inspired by the lightweight philosophy of libraries like `stb`, Watt aims to deliver minimal, efficient, and reusable components for building simulation systems.

## Features

- **Header-only**: Just include the headers, no builds, no extra setup.
- **Eigen-powered**: All numerical operations rely on [Eigen](https://eigen.tuxfamily.org/) for efficient linear algebra.
- **Derivative-checked**: Automatic verification of gradients and Hessians using [TinyAD](https://github.com/patr-schm/TinyAD).
- ️**Modular**: Use only what you need. Each header is fully self-contained. You can use any module without depending on the others.

##  Modules

Currently available modules include:

- **Deformation Gradient**
  - Compute deformation gradients for tetrahedral & triangle meshes.
  - Jacobian of the deformation gradient with respect to nodal positions.
  - Tetrahedral volume computation.

- **Kronecker Product**
  - Kronecker product implementation for fixed-size matrices in Eigen.
  - [Perfect Shuffle Matrix or Commutation Matrix](https://en.wikipedia.org/wiki/Commutation_matrix)

- **Rotation-Variant SVD and Polar Decomposition**

- **Spring Energy**

- **Stable Neo-Hookean Energy**
