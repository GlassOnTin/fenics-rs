//! High-level Poisson problem solver and error norm computation.

use crate::cg::solve_cg;
use fenics_assembly::poisson::{
    apply_dirichlet_bc, assemble_rhs_2d, assemble_rhs_3d, assemble_stiffness_2d,
    assemble_stiffness_3d, boundary_dofs_2d, boundary_dofs_3d, DirichletBC,
};
use fenics_element::{
    jacobian::AffineSimplexMap,
    lagrange::{FiniteElement, P1Tetrahedron, P1Triangle},
    quadrature::{tetrahedron_quadrature, triangle_quadrature},
};
use fenics_mesh::{TetrahedronMesh, TriangleMesh};

/// Result of a Poisson solve.
pub struct PoissonSolution {
    /// Solution vector values at each nodal DOF
    pub u: Vec<f64>,
    /// Number of solver iterations
    pub iterations: usize,
    /// Final residual norm
    pub residual: f64,
}

/// Solve the 2D Poisson problem $-\Delta u = f$ with homogeneous or non-homogeneous Dirichlet BCs.
pub fn solve_poisson_2d<F, G>(
    mesh: &TriangleMesh,
    source: F,
    boundary_value: G,
    tol: f64,
    max_iter: usize,
) -> Result<PoissonSolution, String>
where
    F: Fn([f64; 2]) -> f64 + Sync + Send,
    G: Fn([f64; 2]) -> f64,
{
    let a_raw = assemble_stiffness_2d(mesh);
    let b_raw = assemble_rhs_2d(mesh, source);

    let bnd_dofs = boundary_dofs_2d(mesh);
    let bnd_values: Vec<f64> = bnd_dofs
        .iter()
        .map(|&idx| boundary_value(mesh.vertices[idx]))
        .collect();

    let bc = DirichletBC {
        dofs: bnd_dofs,
        values: bnd_values,
    };

    let (a, b) = apply_dirichlet_bc(&a_raw, &b_raw, &bc);
    let (u, iterations, residual) = solve_cg(&a, &b, None, tol, max_iter)?;

    Ok(PoissonSolution {
        u,
        iterations,
        residual,
    })
}

/// Compute L2 norm of the error: ||u_h - u_exact||_{L2} = sqrt( \int_Omega (u_h - u_exact)^2 dx )
pub fn compute_l2_error_2d<F>(mesh: &TriangleMesh, u_h: &[f64], u_exact: F) -> f64
where
    F: Fn([f64; 2]) -> f64,
{
    let elem = P1Triangle;
    let quad = triangle_quadrature(3); // Higher order quadrature for accurate error norm

    let mut total_sq_error: f64 = 0.0;

    for cell in &mesh.cells {
        let verts = [
            mesh.vertices[cell[0]],
            mesh.vertices[cell[1]],
            mesh.vertices[cell[2]],
        ];
        let map = AffineSimplexMap::from_triangle_vertices(&verts).unwrap();
        let det_j = map.det_jacobian.abs();

        let u_cell = [u_h[cell[0]], u_h[cell[1]], u_h[cell[2]]];

        for q in &quad.points {
            let x_phys = map.map_to_physical(&verts[0], &q.point);
            let exact_val = u_exact(x_phys);

            let basis_vals = elem.evaluate_basis(&q.point);
            let approx_val =
                u_cell[0] * basis_vals[0] + u_cell[1] * basis_vals[1] + u_cell[2] * basis_vals[2];

            let err = approx_val - exact_val;
            total_sq_error += q.weight * det_j * err * err;
        }
    }

    total_sq_error.sqrt()
}

/// Solve the 3D Poisson problem $-\Delta u = f$ with Dirichlet BCs.
pub fn solve_poisson_3d<F, G>(
    mesh: &TetrahedronMesh,
    source: F,
    boundary_value: G,
    tol: f64,
    max_iter: usize,
) -> Result<PoissonSolution, String>
where
    F: Fn([f64; 3]) -> f64 + Sync + Send,
    G: Fn([f64; 3]) -> f64,
{
    let a_raw = assemble_stiffness_3d(mesh);
    let b_raw = assemble_rhs_3d(mesh, source);

    let bnd_dofs = boundary_dofs_3d(mesh);
    let bnd_values: Vec<f64> = bnd_dofs
        .iter()
        .map(|&idx| boundary_value(mesh.vertices[idx]))
        .collect();

    let bc = DirichletBC {
        dofs: bnd_dofs,
        values: bnd_values,
    };

    let (a, b) = apply_dirichlet_bc(&a_raw, &b_raw, &bc);
    let (u, iterations, residual) = solve_cg(&a, &b, None, tol, max_iter)?;

    Ok(PoissonSolution {
        u,
        iterations,
        residual,
    })
}

/// Compute L2 norm of the error in 3D: ||u_h - u_exact||_{L2}
pub fn compute_l2_error_3d<F>(mesh: &TetrahedronMesh, u_h: &[f64], u_exact: F) -> f64
where
    F: Fn([f64; 3]) -> f64,
{
    let elem = P1Tetrahedron;
    let quad = tetrahedron_quadrature(3);

    let mut total_sq_error: f64 = 0.0;

    for cell in &mesh.cells {
        let verts = [
            mesh.vertices[cell[0]],
            mesh.vertices[cell[1]],
            mesh.vertices[cell[2]],
            mesh.vertices[cell[3]],
        ];
        let map = AffineSimplexMap::from_tetrahedron_vertices(&verts).unwrap();
        let det_j = map.det_jacobian.abs();

        let u_cell = [u_h[cell[0]], u_h[cell[1]], u_h[cell[2]], u_h[cell[3]]];

        for q in &quad.points {
            let x_phys = map.map_to_physical(&verts[0], &q.point);
            let exact_val = u_exact(x_phys);

            let basis_vals = elem.evaluate_basis(&q.point);
            let approx_val = u_cell[0] * basis_vals[0]
                + u_cell[1] * basis_vals[1]
                + u_cell[2] * basis_vals[2]
                + u_cell[3] * basis_vals[3];

            let err = approx_val - exact_val;
            total_sq_error += q.weight * det_j * err * err;
        }
    }

    total_sq_error.sqrt()
}
