//! Nonlinear Neo-Hookean Hyperelasticity solver with large deformations.
//!
//! Solves the finite-strain elasticity boundary value problem:
//!   div(P) + f = 0
//! where P is the first Piola-Kirchhoff stress tensor derived from the
//! compressible Neo-Hookean strain energy function:
//!   psi(F) = (mu/2) * (tr(C) - d) - mu * ln(J) + (lambda/2) * (ln(J))^2
//! using exact tangent elasticity tensors and Newton-Raphson iterations.

use crate::newton::{NewtonResult, NewtonSolver};
use faer::sparse::SparseColMat;
use fenics_element::jacobian::AffineSimplexMap;
use fenics_element::lagrange::{FiniteElement, P1Triangle};
use fenics_mesh::TriangleMesh;

/// Material properties for compressible Neo-Hookean hyperelasticity.
#[derive(Clone, Copy, Debug)]
pub struct NeoHookeanMaterial {
    /// Shear modulus mu
    pub mu: f64,
    /// First Lamé parameter lambda
    pub lambda: f64,
}

impl NeoHookeanMaterial {
    pub fn from_youngs_poisson(e: f64, nu: f64) -> Self {
        let mu = e / (2.0 * (1.0 + nu));
        let lambda = (e * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Self { mu, lambda }
    }
}

/// Boundary conditions for hyperelastic problem.
#[derive(Clone, Debug)]
pub struct HyperelasticBC {
    /// Clamped / prescribed DOFs: (global_dof_idx, value)
    pub dirichlet_dofs: Vec<(usize, f64)>,
}

/// Compute the 2D deformation gradient F = I + grad(u).
#[inline]
pub fn deformation_gradient_2d(grad_u: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [1.0 + grad_u[0][0], grad_u[0][1]],
        [grad_u[1][0], 1.0 + grad_u[1][1]],
    ]
}

/// Compute determinant and inverse transpose of a 2x2 matrix.
#[inline]
pub fn det_inv_transpose_2d(f: &[[f64; 2]; 2]) -> (f64, [[f64; 2]; 2]) {
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let inv_t = [
        [f[1][1] / det, -f[1][0] / det],
        [-f[0][1] / det, f[0][0] / det],
    ];
    (det, inv_t)
}

/// First Piola-Kirchhoff stress tensor P for Neo-Hookean material.
pub fn first_piola_kirchhoff_2d(
    mat: &NeoHookeanMaterial,
    f: &[[f64; 2]; 2],
) -> ([[f64; 2]; 2], f64) {
    let (j, f_inv_t) = det_inv_transpose_2d(f);
    let ln_j = j.ln();

    let mut p = [[0.0; 2]; 2];
    for i in 0..2 {
        for j_idx in 0..2 {
            p[i][j_idx] =
                mat.mu * (f[i][j_idx] - f_inv_t[i][j_idx]) + mat.lambda * ln_j * f_inv_t[i][j_idx];
        }
    }
    (p, j)
}

/// Tangent elasticity tensor tangent dP_ij / dF_kl (4x4 in 2D).
pub fn tangent_moduli_2d(mat: &NeoHookeanMaterial, f: &[[f64; 2]; 2]) -> [[[[f64; 2]; 2]; 2]; 2] {
    let (j, f_inv_t) = det_inv_transpose_2d(f);
    let ln_j = j.ln();

    let mut c = [[[[0.0; 2]; 2]; 2]; 2];
    for i in 0..2 {
        for j_idx in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let delta_ik = if i == k { 1.0 } else { 0.0 };
                    let delta_jl = if j_idx == l { 1.0 } else { 0.0 };

                    let term1 = mat.mu * delta_ik * delta_jl;
                    let term2 = (mat.mu - mat.lambda * ln_j) * f_inv_t[i][l] * f_inv_t[k][j_idx];
                    let term3 = mat.lambda * f_inv_t[i][j_idx] * f_inv_t[k][l];

                    c[i][j_idx][k][l] = term1 + term2 + term3;
                }
            }
        }
    }
    c
}

/// Solve 2D hyperelastic boundary value problem using Newton-Raphson.
#[allow(clippy::needless_range_loop)]
pub fn solve_hyperelastic_2d(
    mesh: &TriangleMesh,
    material: &NeoHookeanMaterial,
    bcs: &HyperelasticBC,
) -> Result<NewtonResult, String> {
    let n_nodes = mesh.num_vertices();
    let total_dofs = 2 * n_nodes;
    let elem = P1Triangle;

    let is_constrained = {
        let mut flags = vec![false; total_dofs];
        for &(dof, _) in &bcs.dirichlet_dofs {
            flags[dof] = true;
        }
        flags
    };

    // Precompute cell geometry maps and physical gradients
    struct CellData {
        nodes: [usize; 3],
        area: f64,
        phys_grads: [[f64; 2]; 3],
    }

    let mut cells_data = Vec::with_capacity(mesh.cells.len());
    let ref_grads = elem.evaluate_gradients(&[0.0, 0.0]);

    for cell in &mesh.cells {
        let verts = [
            mesh.vertices[cell[0]],
            mesh.vertices[cell[1]],
            mesh.vertices[cell[2]],
        ];
        if let Ok(map) = AffineSimplexMap::from_triangle_vertices(&verts) {
            let area = 0.5 * map.det_jacobian.abs();
            let mut phys_grads = [[0.0, 0.0]; 3];
            for i in 0..3 {
                phys_grads[i] = map.transform_gradient(&ref_grads[i]);
            }
            cells_data.push(CellData {
                nodes: *cell,
                area,
                phys_grads,
            });
        }
    }

    // Residual evaluation closure R(u)
    let residual_fn = |u: &[f64]| -> Vec<f64> {
        let mut r = vec![0.0; total_dofs];

        for cdata in &cells_data {
            // Compute displacement gradient in cell: grad_u[dim][coord] = sum_a u_{dim, a} * dphi_a / dX_{coord}
            let mut grad_u = [[0.0; 2]; 2];
            for a in 0..3 {
                let node = cdata.nodes[a];
                let u_x = u[node];
                let u_y = u[node + n_nodes];
                let dphi = &cdata.phys_grads[a];

                grad_u[0][0] += u_x * dphi[0];
                grad_u[0][1] += u_x * dphi[1];
                grad_u[1][0] += u_y * dphi[0];
                grad_u[1][1] += u_y * dphi[1];
            }

            let f = deformation_gradient_2d(&grad_u);
            let (p, _j) = first_piola_kirchhoff_2d(material, &f);

            // Virtual work: \int P : grad(v) dOmega
            for a in 0..3 {
                let node = cdata.nodes[a];
                let dphi = &cdata.phys_grads[a];

                let mut fx = 0.0;
                let mut fy = 0.0;
                for j_coord in 0..2 {
                    fx += p[0][j_coord] * dphi[j_coord];
                    fy += p[1][j_coord] * dphi[j_coord];
                }

                r[node] += fx * cdata.area;
                r[node + n_nodes] += fy * cdata.area;
            }
        }

        // Zero out residual on constrained DOFs
        for &(dof, val) in &bcs.dirichlet_dofs {
            r[dof] = u[dof] - val;
        }

        r
    };

    // Jacobian tangent stiffness closure J(u)
    let jacobian_fn = |u: &[f64]| -> SparseColMat<usize, f64> {
        let mut triplets = Vec::new();

        for cdata in &cells_data {
            let mut grad_u = [[0.0; 2]; 2];
            for a in 0..3 {
                let node = cdata.nodes[a];
                let u_x = u[node];
                let u_y = u[node + n_nodes];
                let dphi = &cdata.phys_grads[a];

                grad_u[0][0] += u_x * dphi[0];
                grad_u[0][1] += u_x * dphi[1];
                grad_u[1][0] += u_y * dphi[0];
                grad_u[1][1] += u_y * dphi[1];
            }

            let f = deformation_gradient_2d(&grad_u);
            let c_mod = tangent_moduli_2d(material, &f);

            // Cell stiffness matrix K_{a,i, b,k} = \int dphi_a/dX_j * C_{ijkl} * dphi_b/dX_l dOmega
            for a in 0..3 {
                let node_a = cdata.nodes[a];
                let dphi_a = &cdata.phys_grads[a];

                for b in 0..3 {
                    let node_b = cdata.nodes[b];
                    let dphi_b = &cdata.phys_grads[b];

                    for i in 0..2 {
                        let dof_row = node_a + i * n_nodes;
                        if is_constrained[dof_row] {
                            continue;
                        }

                        for k in 0..2 {
                            let dof_col = node_b + k * n_nodes;
                            if is_constrained[dof_col] {
                                continue;
                            }

                            let mut kab = 0.0;
                            for j_coord in 0..2 {
                                for l_coord in 0..2 {
                                    kab += dphi_a[j_coord]
                                        * c_mod[i][j_coord][k][l_coord]
                                        * dphi_b[l_coord];
                                }
                            }

                            triplets.push((dof_row, dof_col, kab * cdata.area));
                        }
                    }
                }
            }
        }

        // Set 1.0 on diagonal for constrained DOFs
        for &(dof, _) in &bcs.dirichlet_dofs {
            triplets.push((dof, dof, 1.0));
        }

        SparseColMat::try_new_from_triplets(total_dofs, total_dofs, &triplets).unwrap()
    };

    let mut u_init = vec![0.0; total_dofs];
    for &(dof, val) in &bcs.dirichlet_dofs {
        u_init[dof] = val;
    }

    let solver = NewtonSolver {
        max_iterations: 30,
        absolute_tolerance: 1e-7,
        relative_tolerance: 1e-6,
        relaxation_factor: 1.0,
        backtracking: true,
    };

    solver.solve(&u_init, residual_fn, jacobian_fn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use fenics_mesh::generators::unit_square;

    #[test]
    fn test_hyperelastic_tensile_stretch() {
        // 2D unit square [0, 1] x [0, 1] clamped at x = 0 and pulled at x = 1 by 10% (delta_x = 0.1)
        let mesh = unit_square(4, 4);
        let n = mesh.num_vertices();

        let mut bcs_dofs = Vec::new();
        let delta_x = 0.10; // 10% elongation

        for i in 0..n {
            let x = mesh.vertices[i][0];
            let y = mesh.vertices[i][1];

            // Clamp left side at x = 0: ux = 0, uy = 0 (pin lower corner to prevent rigid shift)
            if x.abs() < 1e-6 {
                bcs_dofs.push((i, 0.0)); // ux = 0
                if y.abs() < 1e-6 {
                    bcs_dofs.push((i + n, 0.0)); // uy = 0
                }
            }

            // Pull right side at x = 1: ux = delta_x
            if (x - 1.0).abs() < 1e-6 {
                bcs_dofs.push((i, delta_x));
            }
        }

        let material = NeoHookeanMaterial::from_youngs_poisson(1000.0, 0.3);
        let bcs = HyperelasticBC {
            dirichlet_dofs: bcs_dofs,
        };

        let res = solve_hyperelastic_2d(&mesh, &material, &bcs).unwrap();

        assert!(res.converged);
        assert!(res.iterations <= 6); // Quadratic Newton convergence in ~3-4 iters

        // Center line nodes at x = 0.5 should have displacement approximately delta_x / 2 = 0.05
        for i in 0..n {
            let x = mesh.vertices[i][0];
            if (x - 0.5).abs() < 1e-6 {
                let ux = res.solution[i];
                assert_relative_eq!(ux, 0.05, epsilon = 0.015);
            }
        }
    }
}
