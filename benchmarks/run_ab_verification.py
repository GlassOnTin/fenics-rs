#!/usr/bin/env python3
"""
DOLFINx 0.10.0 Reference Benchmark Runner for fenics-rs A/B Verification.
Computes reference FE solutions for Poisson, Elasticity, Heat, and Stokes problems,
exporting metrics to JSON for automated comparison against fenics-rs.
"""

import json
import time
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

def benchmark_poisson_2d():
    print("[DOLFINx] Running Benchmark 1: 2D Poisson manufactured solution...")
    comm = MPI.COMM_WORLD
    nx = 16
    ny = 16
    domain = mesh.create_unit_square(comm, nx, ny, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Boundary condition u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(fem.Constant(domain, 0.0), boundary_dofs, V)

    # Variational form: a = \int \grad u \cdot \grad v dx, L = \int f v dx
    # with f(x, y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    t0 = time.perf_counter()
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="poisson_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # L2 error vs exact solution u_exact = sin(pi*x) * sin(pi*y)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    error_form = fem.form((uh - u_exact)**2 * ufl.dx)
    l2_error = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))

    u_center = uh.x.array[domain.geometry.x[:, 0].argsort()][len(uh.x.array) // 2]
    max_val = np.max(uh.x.array)

    print(f"  DOLFINx 2D Poisson: L2 Error = {l2_error:.6e}, Max = {max_val:.6f}, Time = {elapsed_ms:.2f} ms")
    return {
        "l2_error": float(l2_error),
        "max_value": float(max_val),
        "elapsed_ms": float(elapsed_ms),
        "num_dofs": int(V.dofmap.index_map.size_global),
    }

def benchmark_cantilever_elasticity():
    print("[DOLFINx] Running Benchmark 2: 3D Cantilever Beam Bending...")
    comm = MPI.COMM_WORLD
    L = 1.0
    W = 0.1
    H = 0.1
    nx, ny, nz = 10, 2, 2
    domain = mesh.create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([L, W, H])], [nx, ny, nz], mesh.CellType.tetrahedron)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    # Clamp at x = 0
    def clamped_boundary(x):
        return np.isclose(x[0], 0.0)

    clamped_dofs = fem.locate_dofs_geometrical(V, clamped_boundary)
    bc = fem.dirichletbc(fem.Constant(domain, np.array([0.0, 0.0, 0.0], dtype=np.float64)), clamped_dofs, V)

    # Elasticity parameters: E = 1e7, nu = 0.3
    E = 1e7
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    # Downward traction at tip: q_y = -1000.0 or body force f = [0, -1000, 0]
    f = fem.Constant(domain, np.array([0.0, -1000.0, 0.0], dtype=np.float64))
    L_form = ufl.dot(f, v) * ufl.dx

    t0 = time.perf_counter()
    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options_prefix="elasticity_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Tip deflection (minimum uy at x ~ L)
    u_vals = uh.x.array.reshape(-1, 3)
    coords = domain.geometry.x
    tip_indices = np.where(np.isclose(coords[:, 0], L))[0]
    tip_uy = np.mean(u_vals[tip_indices, 1])

    print(f"  DOLFINx 3D Elasticity: Tip Deflection = {tip_uy:.6f} m, Time = {elapsed_ms:.2f} ms")
    return {
        "tip_uy": float(tip_uy),
        "elapsed_ms": float(elapsed_ms),
        "num_dofs": int(V.dofmap.index_map.size_global * 3),
    }

def main():
    results = {
        "dolfinx_version": dolfinx.__version__,
        "poisson_2d": benchmark_poisson_2d(),
        "elasticity_3d": benchmark_cantilever_elasticity(),
    }
    with open("benchmarks/dolfinx_reference.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[DOLFINx] Reference benchmarks dumped to benchmarks/dolfinx_reference.json successfully!")

if __name__ == "__main__":
    main()
