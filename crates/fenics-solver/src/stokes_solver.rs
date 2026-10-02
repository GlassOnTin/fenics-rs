//! Incompressible Stokes and Navier-Stokes flow solver.
//!
//! Solves the coupled saddle-point Stokes system:
//!   -mu * \nabla^2 u + \nabla p = f
//!   \nabla \cdot u = 0
//! using mixed finite elements and direct LU decomposition.

use crate::direct::solve_direct;
use faer::sparse::SparseColMat;
use fenics_element::jacobian::AffineSimplexMap;
use fenics_element::lagrange::{FiniteElement, P1Triangle};
use fenics_element::quadrature::triangle_quadrature;
use fenics_mesh::TriangleMesh;

/// Stokes flow solution containing velocity components and pressure.
#[derive(Clone, Debug)]
pub struct StokesSolution {
    pub ux: Vec<f64>,
    pub uy: Vec<f64>,
    pub pressure: Vec<f64>,
}

/// Boundary condition specification for Stokes flow.
#[derive(Clone, Debug)]
pub struct StokesBC {
    /// DOFs where ux is specified
    pub ux_dofs: Vec<(usize, f64)>,
    /// DOFs where uy is specified
    pub uy_dofs: Vec<(usize, f64)>,
    /// DOFs where pressure is pinned (to fix hydrostatic nullspace)
    pub p_dofs: Vec<(usize, f64)>,
}

/// Solve 2D Stokes equations on a triangle mesh.
pub fn solve_stokes_2d(
    mesh: &TriangleMesh,
    viscosity: f64,
    bcs: &StokesBC,
) -> Result<StokesSolution, String> {
    let n_nodes = mesh.num_vertices();
    let elem = P1Triangle;
    let quad = triangle_quadrature(2);

    // Global layout:
    // [0 .. n_nodes)             : ux
    // [n_nodes .. 2*n_nodes)     : uy
    // [2*n_nodes .. 3*n_nodes)   : p
    let total_dofs = 3 * n_nodes;
    let mut triplets = Vec::new();
    let mut rhs = vec![0.0; total_dofs];

    // Assembly loop over cells
    for cell in &mesh.cells {
        let verts = [
            mesh.vertices[cell[0]],
            mesh.vertices[cell[1]],
            mesh.vertices[cell[2]],
        ];
        let map = match AffineSimplexMap::from_triangle_vertices(&verts) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let det_j = map.det_jacobian.abs();
        let ref_grads = elem.evaluate_gradients(&[0.0, 0.0]);
        let mut phys_grads = [[0.0, 0.0]; 3];
        for i in 0..3 {
            phys_grads[i] = map.transform_gradient(&ref_grads[i]);
        }

        let area = 0.5 * det_j;

        // 1. Viscous block A: mu * grad(u) : grad(v)
        for i in 0..3 {
            let row_x = cell[i];
            let row_y = cell[i] + n_nodes;
            for j in 0..3 {
                let col_x = cell[j];
                let col_y = cell[j] + n_nodes;
                let dot_g =
                    phys_grads[i][0] * phys_grads[j][0] + phys_grads[i][1] * phys_grads[j][1];
                let val = viscosity * dot_g * area;

                triplets.push((row_x, col_x, val));
                triplets.push((row_y, col_y, val));
            }
        }

        // 2. Divergence and Gradient coupling B and B^T
        // B_kj = -\int p_k * div(u_j) dx = -\int p_k (dux_j/dx + duy_j/dy) dx
        // B^T_jk = -\int q_k * div(v_j) dx
        for q_pt in &quad.points {
            let phi = elem.evaluate_basis(&q_pt.point);
            let w_det = q_pt.weight * det_j;

            for i in 0..3 {
                let row_x = cell[i];
                let row_y = cell[i] + n_nodes;
                let dphi_x = phys_grads[i][0];
                let dphi_y = phys_grads[i][1];

                for k in 0..3 {
                    let col_p = cell[k] + 2 * n_nodes;
                    let psi = phi[k];

                    // Gradient term in momentum equations (-p * div(v))
                    let bx = -psi * dphi_x * w_det;
                    let by = -psi * dphi_y * w_det;

                    triplets.push((row_x, col_p, bx));
                    triplets.push((row_y, col_p, by));

                    // Divergence term in continuity equation (-div(u) * q)
                    triplets.push((col_p, row_x, bx));
                    triplets.push((col_p, row_y, by));
                }
            }
        }

        // 3. Pressure stabilization term (PSPG): -tau * grad(p) . grad(q)
        let tau = 0.005;
        for i in 0..3 {
            let row_p = cell[i] + 2 * n_nodes;
            for j in 0..3 {
                let col_p = cell[j] + 2 * n_nodes;
                let dot_g =
                    phys_grads[i][0] * phys_grads[j][0] + phys_grads[i][1] * phys_grads[j][1];
                triplets.push((row_p, col_p, -tau * dot_g * area));
            }
        }
    }

    // Apply Dirichlet boundary conditions via row/col zeroing and diagonal identity
    let mut is_constrained = vec![false; total_dofs];
    let mut bc_values = vec![0.0; total_dofs];

    for &(dof, val) in &bcs.ux_dofs {
        is_constrained[dof] = true;
        bc_values[dof] = val;
    }
    for &(dof, val) in &bcs.uy_dofs {
        let global_dof = dof + n_nodes;
        is_constrained[global_dof] = true;
        bc_values[global_dof] = val;
    }
    for &(dof, val) in &bcs.p_dofs {
        let global_dof = dof + 2 * n_nodes;
        is_constrained[global_dof] = true;
        bc_values[global_dof] = val;
    }

    let mut filtered_triplets = Vec::new();
    for (r, c, v) in triplets {
        if is_constrained[r] {
            // Row is constrained, will be set to identity below
        } else if is_constrained[c] {
            rhs[r] -= v * bc_values[c];
        } else {
            filtered_triplets.push((r, c, v));
        }
    }

    // Set 1.0 on diagonal for all constrained DOFs and set RHS
    for i in 0..total_dofs {
        if is_constrained[i] {
            filtered_triplets.push((i, i, 1.0));
            rhs[i] = bc_values[i];
        }
    }

    let system_mat =
        SparseColMat::try_new_from_triplets(total_dofs, total_dofs, &filtered_triplets).unwrap();
    let sol = solve_direct(&system_mat, &rhs)?;

    let ux = sol[0..n_nodes].to_vec();
    let uy = sol[n_nodes..2 * n_nodes].to_vec();
    let pressure = sol[2 * n_nodes..3 * n_nodes].to_vec();

    Ok(StokesSolution { ux, uy, pressure })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use fenics_mesh::generators::unit_square;

    #[test]
    fn test_poiseuille_channel_flow() {
        // Poiseuille flow in a 2D channel [0, 1] x [0, 1]:
        // Inflow: parabolic profile ux(0, y) = 4 * u_max * y * (1 - y) with u_max = 1.0
        // Top and bottom walls (y=0, y=1): no-slip ux=uy=0
        // Pinned pressure at outlet (x=1): p = 0.0
        let mesh = unit_square(8, 8);
        let n = mesh.num_vertices();

        let mut ux_bcs = Vec::new();
        let mut uy_bcs = Vec::new();
        let mut p_bcs = Vec::new();

        for i in 0..n {
            let x = mesh.vertices[i][0];
            let y = mesh.vertices[i][1];

            // Top and bottom walls: no-slip
            if y.abs() < 1e-6 || (y - 1.0).abs() < 1e-6 {
                ux_bcs.push((i, 0.0));
                uy_bcs.push((i, 0.0));
            }
            // Inlet (x=0)
            else if x.abs() < 1e-6 {
                let u_inlet = 4.0 * y * (1.0 - y);
                ux_bcs.push((i, u_inlet));
                uy_bcs.push((i, 0.0));
            }

            // Pin pressure at outlet (x=1)
            if (x - 1.0).abs() < 1e-6 {
                p_bcs.push((i, 0.0));
            }
        }

        let bcs = StokesBC {
            ux_dofs: ux_bcs,
            uy_dofs: uy_bcs,
            p_dofs: p_bcs,
        };

        let viscosity = 1.0;
        let sol = solve_stokes_2d(&mesh, viscosity, &bcs).unwrap();

        // Check center channel velocity at y = 0.5: parabolic peak should be ~1.0
        for i in 0..n {
            let x = mesh.vertices[i][0];
            let y = mesh.vertices[i][1];
            if (y - 0.5).abs() < 1e-6 && x > 0.1 && x < 0.9 {
                assert_relative_eq!(sol.ux[i], 1.0, epsilon = 0.20);
                assert_relative_eq!(sol.uy[i], 0.0, epsilon = 0.10);
            }
        }
    }
}
