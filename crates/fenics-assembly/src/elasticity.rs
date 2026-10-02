//! 3D Linear Elasticity (Navier-Cauchy) stiffness assembly and stress tensor recovery.

use faer::sparse::SparseColMat;
use faer::Unbind;
use fenics_element::{
    jacobian::AffineSimplexMap,
    lagrange::{FiniteElement, P1Tetrahedron},
};
use fenics_mesh::TetrahedronMesh;
use rayon::prelude::*;

/// Isotropic linear elastic material properties.
#[derive(Clone, Copy, Debug)]
pub struct ElasticMaterial {
    /// Young's modulus E (in Pascals)
    pub youngs_modulus: f64,
    /// Poisson's ratio nu (dimensionless, typically 0.2 - 0.35)
    pub poissons_ratio: f64,
}

impl ElasticMaterial {
    pub fn new(youngs_modulus: f64, poissons_ratio: f64) -> Self {
        assert!(
            poissons_ratio > -1.0 && poissons_ratio < 0.5,
            "Poisson's ratio must be in (-1.0, 0.5) for physical stability"
        );
        Self {
            youngs_modulus,
            poissons_ratio,
        }
    }

    /// First Lamé parameter lambda = (E * nu) / ((1 + nu) * (1 - 2*nu))
    #[inline]
    pub fn lame_lambda(&self) -> f64 {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;
        (e * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// Second Lamé parameter (Shear modulus) mu = E / (2 * (1 + nu))
    #[inline]
    pub fn lame_mu(&self) -> f64 {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;
        e / (2.0 * (1.0 + nu))
    }
}

/// Stress state of an element or point.
#[derive(Clone, Copy, Debug, Default)]
pub struct StressState {
    /// Cauchy stress tensor sigma (symmetric 3x3)
    pub stress: [[f64; 3]; 3],
    /// Infinitesimal strain tensor epsilon (symmetric 3x3)
    pub strain: [[f64; 3]; 3],
    /// Von Mises equivalent scalar stress
    pub von_mises: f64,
    /// Trace of stress (hydrostatic pressure = -trace/3)
    pub trace: f64,
}

/// Dirichlet boundary condition for vector displacement fields.
#[derive(Clone, Debug)]
pub struct VectorDirichletBC {
    /// List of (dof_index, prescribed_value).
    /// For vertex v: dof 3*v is u_x, 3*v + 1 is u_y, 3*v + 2 is u_z.
    pub prescribed_dofs: Vec<(usize, f64)>,
}

/// Assemble the 3D linear elasticity stiffness matrix K.
/// Total DOFs = 3 * mesh.num_vertices().
pub fn assemble_elasticity_stiffness_3d(
    mesh: &TetrahedronMesh,
    material: &ElasticMaterial,
) -> SparseColMat<usize, f64> {
    let elem = P1Tetrahedron;
    let n_verts = mesh.num_vertices();
    let n_dofs = 3 * n_verts;

    let lambda = material.lame_lambda();
    let mu = material.lame_mu();

    let ref_grads = elem.evaluate_gradients(&[0.0, 0.0, 0.0]);

    let triplets: Vec<(usize, usize, f64)> = mesh
        .cells
        .par_iter()
        .flat_map(|cell| {
            let verts = [
                mesh.vertices[cell[0]],
                mesh.vertices[cell[1]],
                mesh.vertices[cell[2]],
                mesh.vertices[cell[3]],
            ];

            let map = match AffineSimplexMap::from_tetrahedron_vertices(&verts) {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };

            let vol = (1.0 / 6.0) * map.det_jacobian.abs();

            let mut g = [[0.0; 3]; 4];
            for i in 0..4 {
                g[i] = map.transform_gradient(&ref_grads[i]);
            }

            // 4 nodes * 3 displacement DOFs = 12 DOFs per tetrahedron
            // Local stiffness matrix is 12x12
            let mut local_triplets = Vec::with_capacity(144);

            for i in 0..4 {
                let gi = g[i];
                let vi = cell[i];

                for j in 0..4 {
                    let gj = g[j];
                    let vj = cell[j];
                    let gi_dot_gj = gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2];

                    for alpha in 0..3 {
                        let dof_row = 3 * vi + alpha;

                        for beta in 0..3 {
                            let dof_col = 3 * vj + beta;

                            let delta_ab = if alpha == beta { 1.0 } else { 0.0 };

                            // K_{i alpha, j beta} = vol * [ lambda * (gj)_beta * (gi)_alpha
                            //                             + mu * (gi . gj) * delta_ab
                            //                             + mu * (gj)_alpha * (gi)_beta ]
                            let k_val = vol
                                * (lambda * gj[beta] * gi[alpha]
                                    + mu * gi_dot_gj * delta_ab
                                    + mu * gj[alpha] * gi[beta]);

                            local_triplets.push((dof_row, dof_col, k_val));
                        }
                    }
                }
            }

            local_triplets
        })
        .collect();

    SparseColMat::try_new_from_triplets(n_dofs, n_dofs, &triplets).unwrap()
}

/// Assemble the 3D body force load vector (e.g. gravity or body acceleration).
/// f_body is [f_x, f_y, f_z] in N/m^3.
pub fn assemble_elasticity_body_force_3d(mesh: &TetrahedronMesh, f_body: [f64; 3]) -> Vec<f64> {
    let n_dofs = 3 * mesh.num_vertices();
    let mut rhs = vec![0.0; n_dofs];

    for cell in &mesh.cells {
        let verts = [
            mesh.vertices[cell[0]],
            mesh.vertices[cell[1]],
            mesh.vertices[cell[2]],
            mesh.vertices[cell[3]],
        ];

        let map = match AffineSimplexMap::from_tetrahedron_vertices(&verts) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let vol = (1.0 / 6.0) * map.det_jacobian.abs();
        let nodal_load = [
            f_body[0] * (vol * 0.25),
            f_body[1] * (vol * 0.25),
            f_body[2] * (vol * 0.25),
        ];

        for &v in cell {
            rhs[3 * v] += nodal_load[0];
            rhs[3 * v + 1] += nodal_load[1];
            rhs[3 * v + 2] += nodal_load[2];
        }
    }

    rhs
}

/// Apply vector Dirichlet boundary conditions to the elasticity system.
pub fn apply_vector_dirichlet_bc(
    mat: &SparseColMat<usize, f64>,
    rhs: &[f64],
    bc: &VectorDirichletBC,
) -> (SparseColMat<usize, f64>, Vec<f64>) {
    let n = mat.nrows();
    let mut is_dirichlet = vec![false; n];
    let mut dirichlet_val = vec![0.0; n];

    for &(dof, val) in &bc.prescribed_dofs {
        is_dirichlet[dof] = true;
        dirichlet_val[dof] = val;
    }

    let mut new_b = rhs.to_vec();
    let mut new_triplets = Vec::new();

    for col in 0..n {
        let row_inds = mat.as_ref().row_indices_of_col(col);
        let values = mat.as_ref().values_of_col(col);

        for (row_idx, &val) in row_inds.zip(values) {
            let row = row_idx.unbound();

            if is_dirichlet[col] && !is_dirichlet[row] {
                new_b[row] -= val * dirichlet_val[col];
            } else if !is_dirichlet[col] && !is_dirichlet[row] {
                new_triplets.push((row, col, val));
            }
        }
    }

    for &(dof, val) in &bc.prescribed_dofs {
        new_triplets.push((dof, dof, 1.0));
        new_b[dof] = val;
    }

    let new_mat = SparseColMat::try_new_from_triplets(n, n, &new_triplets).unwrap();
    (new_mat, new_b)
}

/// Compute the Cauchy stress tensor, strain tensor, and Von Mises stress for each tetrahedron element.
pub fn compute_element_stresses(
    mesh: &TetrahedronMesh,
    u: &[f64],
    material: &ElasticMaterial,
) -> Vec<StressState> {
    let elem = P1Tetrahedron;
    let lambda = material.lame_lambda();
    let mu = material.lame_mu();
    let ref_grads = elem.evaluate_gradients(&[0.0, 0.0, 0.0]);

    mesh.cells
        .iter()
        .map(|cell| {
            let verts = [
                mesh.vertices[cell[0]],
                mesh.vertices[cell[1]],
                mesh.vertices[cell[2]],
                mesh.vertices[cell[3]],
            ];

            let map = match AffineSimplexMap::from_tetrahedron_vertices(&verts) {
                Ok(m) => m,
                Err(_) => return StressState::default(),
            };

            let mut g = [[0.0; 3]; 4];
            for i in 0..4 {
                g[i] = map.transform_gradient(&ref_grads[i]);
            }

            // Displacement vectors at 4 vertices
            let u_nodes = [
                [u[3 * cell[0]], u[3 * cell[0] + 1], u[3 * cell[0] + 2]],
                [u[3 * cell[1]], u[3 * cell[1] + 1], u[3 * cell[1] + 2]],
                [u[3 * cell[2]], u[3 * cell[2] + 1], u[3 * cell[2] + 2]],
                [u[3 * cell[3]], u[3 * cell[3] + 1], u[3 * cell[3] + 2]],
            ];

            // grad(u)_{i, j} = du_i / dx_j = sum_{node=0}^3 u_{node, i} * g_{node}[j]
            let mut grad_u = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for node in 0..4 {
                        grad_u[i][j] += u_nodes[node][i] * g[node][j];
                    }
                }
            }

            // Strain tensor epsilon = 0.5 * (grad_u + grad_u^T)
            let mut strain = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    strain[i][j] = 0.5 * (grad_u[i][j] + grad_u[j][i]);
                }
            }

            let tr_eps = strain[0][0] + strain[1][1] + strain[2][2];

            // Cauchy stress: sigma = lambda * tr(eps) * I + 2 * mu * eps
            let mut stress = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    stress[i][j] = lambda * tr_eps * delta + 2.0 * mu * strain[i][j];
                }
            }

            // Von Mises stress:
            // sqrt( 0.5 * ( (s00 - s11)^2 + (s11 - s22)^2 + (s22 - s00)^2 + 6*(s01^2 + s12^2 + s20^2) ) )
            let s00 = stress[0][0];
            let s11 = stress[1][1];
            let s22 = stress[2][2];
            let s01 = stress[0][1];
            let s12 = stress[1][2];
            let s20 = stress[2][0];

            let vm_sq = 0.5
                * ((s00 - s11).powi(2)
                    + (s11 - s22).powi(2)
                    + (s22 - s00).powi(2)
                    + 6.0 * (s01.powi(2) + s12.powi(2) + s20.powi(2)));

            StressState {
                stress,
                strain,
                von_mises: vm_sq.max(0.0).sqrt(),
                trace: s00 + s11 + s22,
            }
        })
        .collect()
}

/// Compute smoothed nodal (vertex) Von Mises stresses by volume-weighted averaging over adjacent elements.
pub fn compute_vertex_von_mises(
    mesh: &TetrahedronMesh,
    element_stresses: &[StressState],
) -> Vec<f64> {
    let n_verts = mesh.num_vertices();
    let mut vertex_vm_sum = vec![0.0; n_verts];
    let mut vertex_vol_sum = vec![0.0; n_verts];

    for (cell_idx, cell) in mesh.cells.iter().enumerate() {
        let verts = [
            mesh.vertices[cell[0]],
            mesh.vertices[cell[1]],
            mesh.vertices[cell[2]],
            mesh.vertices[cell[3]],
        ];

        let map = match AffineSimplexMap::from_tetrahedron_vertices(&verts) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let vol = (1.0 / 6.0) * map.det_jacobian.abs();
        let vm = element_stresses[cell_idx].von_mises;

        for &v in cell {
            vertex_vm_sum[v] += vm * vol;
            vertex_vol_sum[v] += vol;
        }
    }

    for i in 0..n_verts {
        if vertex_vol_sum[i] > 1e-15 {
            vertex_vm_sum[i] /= vertex_vol_sum[i];
        }
    }

    vertex_vm_sum
}
