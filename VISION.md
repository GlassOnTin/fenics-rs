# VISION.md: The fenics-rs Project

> **A modern, pure-Rust reimagining of the FEniCS variational finite element framework, engineered for zero-cost abstractions, fearless concurrency, and in-browser scientific simulation via WebAssembly.**

---

## 1. Executive Summary & Motivation

The **FEniCS Project** revolutionized computational science by demonstrating that partial differential equations (PDEs) should be expressed in high-level mathematical notation—the variational weak form—rather than hand-coded element loops. In FEniCS, an engineer writes mathematics; the computer generates the solver.

However, the traditional scientific computing stack has grown increasingly fragile:
* **The Four-Language Tower:** A brittle chain of Python (UFL) calling code generators (FFCx) that invoke host C compilers (`gcc`/`clang`) via subprocesses, loading dynamic shared libraries (`.so`) through CFFI into native C++ (DOLFINx), wrapping C/Fortran solvers (PETSc/BLAS).
* **Deployment Friction:** Installing the stack often mandates multi-gigabyte Docker containers or complex Conda environments.
* **The Sandbox Barrier:** Because it relies on runtime host compilation (`/usr/bin/gcc`) and distributed MPI architectures, traditional FEniCS cannot run inside client-side sandboxes such as web browsers or mobile operating systems.

**fenics-rs** exists to solve this. It migrates the abstract, variational beauty of FEniCS to idiomatic **Rust**. By replacing runtime C generation with compile-time Rust procedural macros, and leveraging native WebAssembly (WASM) compilation, `fenics-rs` turns high-order finite element analysis into a single-binary, cross-platform tool that runs everywhere from HPC servers to a mobile browser.

---

## 2. Core Architectural Pillars

### Pillar I: Compile-Time Variational Calculus
Instead of interpreting weak forms in Python and shelling out to Clang at runtime, `fenics-rs` uses Rust procedural macros (`form!`) to parse variational equations at compile time.
* Expressions such as `inner(grad(u), grad(v)) * dx` expand directly into optimized, SIMD-vectorized element quadrature loops during standard `cargo build`.
* Zero runtime compilation latency. Zero dependency on system C compilers.

### Pillar II: Fearless Concurrent Assembly
Global matrix assembly is historically prone to cache contention and data races when accumulating local element stiffness matrices into global sparse storage.
* By leveraging Rust’s `Send` and `Sync` invariants and work-stealing parallelism (`rayon`), mesh assembly scales across all CPU cores with guaranteed data-race freedom.

### Pillar III: Native WebAssembly & In-Browser Execution
* Rust treats WebAssembly as a first-class tier-1 target.
* The entire simulation engine—from mesh ingestion to sparse linear solves—compiles to a lightweight `.wasm` module.
* Runs in Chrome, Firefox, Safari, and Android without cloud dependencies, server costs, or network latency.

### Pillar IV: Dimension-Invariance via Const Generics
* Meshes, elements, and tensor operations are parameterized over spatial dimension (`const DIM: usize`).
* 1D, 2D, and 3D formulations share the exact same algorithmic codebase without runtime branching or virtual dispatch penalties.

---

## 3. Subsystem Architecture

The repository is structured as a modular Cargo workspace:

```
fenics-rs/
├── crates/
│   ├── fenics-mesh       # Simplicial meshes (triangle/tetrahedron), facet tagging, Gmsh/STL I/O
│   ├── fenics-element    # Reference simplices, Lagrange (P1/P2) bases, Gaussian quadrature
│   ├── fenics-form       # Compile-time variational DSL and symbolic AST (grad, div, inner, dx, ds)
│   ├── fenics-assembly   # Parallel CSR sparse matrix assembly, Dirichlet/Neumann enforcement
│   ├── fenics-solver     # Direct & iterative sparse solvers (powered by faer-rs), Newton solvers
│   └── fenics-wasm       # WebAssembly bridge, JS/TS bindings for WebGL & WebGPU visualizers
├── examples/             # Canonical benchmarks (Poisson, Linear Elasticity, Helmholtz acoustics)
├── LICENSE               # GNU Affero General Public License v3.0
└── README.md
```

### Technical Specs & Math Capabilities
* **Mesh & Topology:** Watertight simplicial topologies (triangles in 2D, tetrahedra in 3D). Boundary facet detection and Dirichlet marker assignments.
* **Basis Functions:** Arbitrary-order Lagrange polynomials (initially focusing on P1 linear and P2 quadratic elements) with exact reference-simplex transformations via Jacobian push-forward/pull-back.
* **Quadrature Rules:** High-precision Gauss-Legendre and Dunavant quadrature tables evaluated at compile time.
* **Linear Algebra:** Pure-Rust high-performance sparse matrix backends via `faer-rs` (Cholesky, LU, and Conjugate Gradient), avoiding external Fortran/C library linking.

---

## 4. Applied Focus: From Pure Geometry to Metamaterials

While `fenics-rs` is a general-purpose PDE engine, its primary design catalyst is **the generative analysis of complex geometric structures**:

1. **Polyhedral Metamaterials:** Evaluating the bulk modulus, shear stiffness, and negative Poisson's ratio (auxetics) of recurring stellation unit cells.
2. **Acoustic & Wave Propagation:** Solving the Helmholtz equation for 3D microphone/speaker arrays and spatial audio transducers.
3. **Biological & Nanoscale Capsids:** Simulating mechanical stability and pressure limits of icosahedral viral capsids and DNA origami cages.
4. **Deployable Origami & Sheet Structures:** Computing localized stress concentrations along polyhedral unfolding creases and hinges.

---

## 5. Development Roadmap

### Phase 1: Mesh, Quadrature, and Reference Elements (Milestone 1)
- [ ] Define `Mesh<const DIM: usize>` with vertex, cell, and facet incidence maps.
- [ ] Implement Gmsh (`.msh`) and watertight surface STL/OBJ mesh loaders.
- [ ] Pre-tabulate Gaussian quadrature points and weights for reference simplices.
- [ ] Implement P1 linear and P2 quadratic shape functions and Jacobian coordinate mappings.
- [ ] Automated tests: Integrate known polynomials over reference tetrahedra to machine precision.

### Phase 2: Variational DSL & Assembly (Milestone 2)
- [ ] Implement the `form!` macro / builder pattern for bilinear forms `a(u, v)` and linear forms `L(v)`.
- [ ] Implement parallel element loop assembly into Compressed Sparse Row (CSR) matrices using `rayon`.
- [ ] Dirichlet boundary condition lifting and condensation.
- [ ] Automated tests: Verify 2D & 3D Poisson equation against analytical manufactured solutions.

### Phase 3: Physics Solvers & Verification (Milestone 3)
- [ ] Integrate `faer-rs` sparse Cholesky and Conjugate Gradient solvers.
- [ ] Implement 3D Linear Elasticity (Navier-Cauchy equations) with Young's modulus and Poisson's ratio tensors.
- [ ] Compute Cauchy and Von Mises stress fields from displacement gradients.
- [ ] Benchmark: Standard cantilever beam deflection test against Euler-Bernoulli and Timoshenko analytical solutions.

### Phase 4: WebAssembly & Interactive Visualization (Milestone 4)
- [ ] Build `fenics-wasm` crate using `wasm-bindgen`.
- [ ] Direct memory transfer of deformed vertices and stress scalar fields to Three.js / WebGPU.
- [ ] End-to-end demo: Generate or load a 3D polyhedral mesh in Chrome, apply simulated load, and render the live stress heatmap.

---

## 6. License & Community

`fenics-rs` is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 
We believe that core scientific computing infrastructure must remain free and open. Any network-accessible simulation services or derivative platforms built upon `fenics-rs` must contribute their source improvements back to the global scientific community.
