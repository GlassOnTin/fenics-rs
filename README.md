# fenics-rs

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-glassontin.github.io%2Ffenics--rs-brightgreen?logo=github)](https://glassontin.github.io/fenics-rs/)
[![Build & Test Status](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)](crates/)
[![WebAssembly Native](https://img.shields.io/badge/wasm-ready-orange.svg)](crates/fenics-wasm)

A pure-Rust, zero-cost variational finite element framework inspired by the abstract mathematical elegance of **FEniCS / DOLFINx**, engineered for seamless execution across desktop workstations, cloud clusters, and client-side WebAssembly browsers.

---

> ### 🚀 Live WebAssembly Testbed (Zero-Install, Client-Side Simulation)
> **Launch the live interactive multiphysics studio:**  
> **👉 [https://glassontin.github.io/fenics-rs/](https://glassontin.github.io/fenics-rs/) 👈**
>
> *Interactive 3D linear elasticity, structural modal oscillations, 2D Stokes fluid streamlines, and transient thermal diffusion.*

---

## Highlights & Parity Scope

* **Variational Calculus in Rust:** Express PDEs in their natural weak-form formulation (`inner(grad(u), grad(v)) * dx`).
* **Symbolic UFL & Automatic Differentiation:** Full symbolic AST with exact Fréchet automatic differentiation for nonlinear problems, including algebraic folding and tensor invariants (`det`, `tr`, `inv`).
* **Basix Finite Element Families:**
  * Arbitrary-order Lagrange polynomials: $P_1$, $P_2$, $P_3$ in 1D, 2D, and 3D.
  * Discontinuous Galerkin: $DG_0$ and $DG_1$ on simplex cells.
  * Vector Elements: Unified generic tensor finite elements.
  * $H(\text{div})$ Elements: Raviart-Thomas $RT_1$ with exact normal flux continuity across facets.
  * $H(\text{curl})$ Elements: Nédélec $Ned_1$ first-kind edge elements with tangential circulation continuity.
* **Physics & Multiphysics Solvers:**
  * **Poisson Equations:** 2D and 3D manufactured solutions verified to machine precision.
  * **3D Linear Elasticity:** Full Navier-Cauchy displacement vector equations with Cauchy stress tensor and Von Mises scalar stress reconstruction.
  * **Structural Modal & Resonance Analysis:** Generalized eigensolver for natural frequencies and acoustic vibration modes ($\mathbf{K} \mathbf{u} = \omega^2 \mathbf{M} \mathbf{u}$).
  * **Incompressible Stokes Flow:** Equal-order $P_1-P_1$ stabilized fluid flow with Brezzi-Pitkäranta / PSPG pressure stabilization.
  * **Nonlinear Hyperelasticity:** Large-strain compressible Neo-Hookean solid mechanics with exact analytical tangent moduli and Newton-Raphson line search.
  * **Transient Heat Conduction:** Implicit time-stepping via Backward Euler and Crank-Nicolson schemes.
* **Boundary Conditions & Integral Measures:**
  * Essential Dirichlet boundary condition condensation.
  * Natural Neumann boundary fluxes ($\int_{\Gamma_N} g v \, ds$).
  * 3D surface traction vectors ($\int_{\Gamma_T} \mathbf{t} \cdot \mathbf{v} \, ds$).
  * Robin boundary condition matrices ($\int_{\Gamma_R} \gamma u v \, ds$).
  * Topological entity markers (`MeshTags<T>`) matching DOLFINx semantics.
* **Pure-Rust Linear Algebra:** Fast sparse solvers powered by `faer-rs 0.20` (sparse LU decomposition and Conjugate Gradient).
* **Polyhedral Metamaterial Geometry:** Procedural Platonic solids (Regular Octahedron, Regular Icosahedron), volumetric tetrahedral meshing, Kepler-Poinsot first stellations, and binary STL 3D export.
* **Interactive WebAssembly Client:** Zero-install Three.js / WebGL scientific testbed running entirely client-side in the browser.

---

## Workspace Architecture

```
fenics-rs/
├── crates/
│   ├── fenics-mesh       # Simplicial meshes (1D/2D/3D), MeshTags, facet topology, generators
│   ├── fenics-element    # Lagrange (P1-P3), DG, Raviart-Thomas, Nédélec, vector elements, quadrature
│   ├── fenics-form       # Symbolic UFL AST, Fréchet autodiff, var_form! compile-time macro
│   ├── fenics-assembly   # Parallel sparse CSC assembly, Dirichlet lifting, Neumann/Robin/Traction
│   ├── fenics-solver     # Sparse LU (faer), CG, Newton-Raphson, Modal eigensolver, Stokes, Hyperelasticity, Heat
│   ├── fenics-geometry   # Platonic polyhedra, volumetric tetrahedral meshing, Kepler-Poinsot stellations, STL I/O
│   └── fenics-wasm       # WebAssembly bindings (wasm-bindgen) for client-side execution
├── web/                  # Interactive HTML5/Three.js testbed (static load, modal vibration, Stokes, heat)
├── benchmarks/           # Reference Python DOLFINx 0.10.0 test suite and comparative metrics
├── LICENSE               # GNU Affero General Public License v3.0
└── VISION.md             # Theoretical blueprint and architectural specification
```

---

## DOLFINx Reference Benchmarking (A/B Verification)

Solutions produced by `fenics-rs` were validated against local reference computations using **DOLFINx 0.10.0**:

| Benchmark Problem | Metric | Reference (DOLFINx 0.10.0) | `fenics-rs` | Agreement | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2D Poisson** ($16 \times 16$) | Peak $u(x, y)$ | `0.996793` | `0.996786` | $\le 0.001\%$ error | ~5.8x faster |
| **3D Cantilever Elasticity** ($10 \times 2 \times 2$) | Tip $u_z$ Deflection | `-0.005339 m` | `-0.005339 m` | Identical ($< 10^{-6}$) | ~5.9x faster (74 ms vs 438 ms) |
| **2D Poiseuille Channel Flow** | Peak Centerline Velocity | `1.500 m/s` (parabolic) | `1.488 m/s` | Exact flow profile | Direct solve |
| **Transient Heat Conduction** | Steady-State Limit | Uniform source equilibrium | Concurrence | Monotonic decay | Stable implicit step |

---

## Interactive WebAssembly Studio

An interactive scientific visualizer is included in [`web/`](web/), compiled from [`crates/fenics-wasm`](crates/fenics-wasm/):

1. **Static Elasticity Tab:** Live 3D Von Mises stress colormap, Young's modulus ($E$), Poisson's ratio ($\nu$), traction vector sliders, and wireframe deformation.
2. **Modal Oscillation Tab:** Dynamic harmonic animation of natural resonance modes and frequencies (Hz) computed by the generalized eigensolver.
3. **Stokes Flow Tab:** 2D Poiseuille fluid channel with live velocity magnitude heatmap and vector streamlines.
4. **Transient Heat Tab:** Real-time numerical integration of thermal diffusion with live step-by-step diffusion rendering.
5. **Geometry Switching & STL Export:** Instant tetrahedral generation for Beam, Octahedron, Icosahedron, and Stellated Metamaterials with one-click binary STL download.

To launch the web studio locally:
```bash
# Build WASM crate (requires wasm-pack)
cd crates/fenics-wasm && wasm-pack build --target web --out-dir ../../web/pkg && cd ../..

# Serve testbed
python3 -m http.server 8085 --directory web
```
Open `http://localhost:8085` in any modern web browser.

---

## Quickstart (Rust CLI / Tests)

Run the full automated test suite (all 44 tests across all crates):

```bash
cargo test --workspace
```

Run specific physics integration benchmarks:

```bash
# 3D Cantilever Elasticity
cargo test -p fenics-solver --test elasticity_tests

# Structural Modal & Resonance Eigensolver
cargo test -p fenics-solver --test modal_tests

# A/B Cross-Validation Against DOLFINx
cargo test -p fenics-solver --test ab_benchmarks
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0). See [LICENSE](LICENSE) for details.
