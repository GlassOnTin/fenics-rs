# fenics-rs

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

A pure-Rust, zero-cost variational finite element framework inspired by the abstract mathematical elegance of **FEniCS**, engineered for seamless execution across desktop, cloud, and in-browser WebAssembly.

---

## Highlights

* **Variational Calculus in Rust:** Express PDEs in their natural weak-form formulation (`inner(grad(u), grad(v)) * dx`).
* **Compile-Time Efficiency:** Employs Rust procedural macros to generate SIMD-vectorized element quadrature loops at compile time, eliminating runtime C compilers (`gcc`/`clang`).
* **Fearless Concurrency:** Race-free parallel global sparse matrix assembly across all CPU cores powered by Rayon.
* **WebAssembly Native:** First-class compilation to WebAssembly for client-side, zero-install physics simulations in Chrome, Firefox, and mobile devices.
* **Pure Rust Linear Algebra:** Backed by high-performance sparse solvers via `faer-rs`.

---

## Vision & Roadmap

See [VISION.md](VISION.md) for the detailed architectural blueprint, mathematical foundations, and milestone plan.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0). See the [LICENSE](LICENSE) file for details.
