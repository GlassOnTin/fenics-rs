//! Variational assembly for Poisson and diffusion equations (2D and 3D).

use faer::sparse::SparseColMat;
use faer::Unbind;
use fenics_element::{
    jacobian::AffineSimplexMap,
    lagrange::{FiniteElement, P1Tetrahedron, P1Triangle},
    quadrature::{tetrahedron_quadrature, triangle_quadrature},
};
use fenics_mesh::{BoundaryFacet, TetrahedronMesh, TriangleMesh};
use rayon::prelude::*;

/// Boundary condition definition for Dirichlet boundaries.
#[derive(Clone, Debug)]
pub struct DirichletBC {
    /// Nodal degrees of freedom where Dirichlet condition is applied
    pub dofs: Vec<usize>,
    /// Prescribed values for the corresponding DOFs
    pub values: Vec<f64>,
}

/// Assemble the 2D stiffness matrix $A_{ij} = \int_\Omega \nabla \phi_i \cdot \nabla \phi_j \, dx$.
pub fn assemble_stiffness_2d(mesh: &TriangleMesh) -> SparseColMat<usize, f64> {
    let elem = P1Triangle;
    let n_dofs = mesh.num_vertices();

    // Reference gradients for P1 triangle:
    // phi0 = 1 - xi - eta -> [-1, -1]
    // phi1 = xi           -> [ 1,  0]
    // phi2 = eta          -> [ 0,  1]
    let ref_grads = elem.evaluate_gradients(&[0.0, 0.0]);

    let triplets: Vec<(usize, usize, f64)> = mesh
        .cells
        .par_iter()
        .flat_map(|cell| {
            let verts = [
                mesh.vertices[cell[0]],
                mesh.vertices[cell[1]],
                mesh.vertices[cell[2]],
            ];
            let map = match AffineSimplexMap::from_triangle_vertices(&verts) {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };

            let area = 0.5 * map.det_jacobian.abs();

            // Compute physical gradients for each of the 3 basis functions:
            let mut phys_grads = [[0.0, 0.0]; 3];
            for i in 0..3 {
                phys_grads[i] = map.transform_gradient(&ref_grads[i]);
            }

            let mut local_triplets = Vec::with_capacity(9);
            for i in 0..3 {
                let gi = phys_grads[i];
                for j in 0..3 {
                    let gj = phys_grads[j];
                    let val = (gi[0] * gj[0] + gi[1] * gj[1]) * area;
                    local_triplets.push((cell[i], cell[j], val));
                }
            }
            local_triplets
        })
        .collect();

    SparseColMat::try_new_from_triplets(n_dofs, n_dofs, &triplets).unwrap()
}

/// Assemble the 2D load vector $b_i = \int_\Omega f(x) \phi_i(x) \, dx$.
pub fn assemble_rhs_2d<F>(mesh: &TriangleMesh, f: F) -> Vec<f64>
where
    F: Fn([f64; 2]) -> f64 + Sync + Send,
{
    let elem = P1Triangle;
    let quad = triangle_quadrature(2);
    let n_dofs = mesh.num_vertices();

    // Parallel accumulation into cell-local vectors, then reduce
    let cell_contributions: Vec<(usize, f64)> = mesh
        .cells
        .par_iter()
        .flat_map(|cell| {
            let verts = [
                mesh.vertices[cell[0]],
                mesh.vertices[cell[1]],
                mesh.vertices[cell[2]],
            ];
            let map = match AffineSimplexMap::from_triangle_vertices(&verts) {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };

            let det_j = map.det_jacobian.abs();
            let mut local_b = [0.0; 3];

            for q in &quad.points {
                let x_phys = map.map_to_physical(&verts[0], &q.point);
                let f_val = f(x_phys);
                let basis_vals = elem.evaluate_basis(&q.point);

                for i in 0..3 {
                    local_b[i] += q.weight * det_j * f_val * basis_vals[i];
                }
            }

            vec![
                (cell[0], local_b[0]),
                (cell[1], local_b[1]),
                (cell[2], local_b[2]),
            ]
        })
        .collect();

    let mut b = vec![0.0; n_dofs];
    for (dof, val) in cell_contributions {
        b[dof] += val;
    }
    b
}

/// Assemble the 3D stiffness matrix $A_{ij} = \int_\Omega \nabla \phi_i \cdot \nabla \phi_j \, dx$.
pub fn assemble_stiffness_3d(mesh: &TetrahedronMesh) -> SparseColMat<usize, f64> {
    let elem = P1Tetrahedron;
    let n_dofs = mesh.num_vertices();
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

            let mut phys_grads = [[0.0, 0.0, 0.0]; 4];
            for i in 0..4 {
                phys_grads[i] = map.transform_gradient(&ref_grads[i]);
            }

            let mut local_triplets = Vec::with_capacity(16);
            for i in 0..4 {
                let gi = phys_grads[i];
                for j in 0..4 {
                    let gj = phys_grads[j];
                    let val = (gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2]) * vol;
                    local_triplets.push((cell[i], cell[j], val));
                }
            }
            local_triplets
        })
        .collect();

    SparseColMat::try_new_from_triplets(n_dofs, n_dofs, &triplets).unwrap()
}

/// Assemble the 3D load vector $b_i = \int_\Omega f(x) \phi_i(x) \, dx$.
pub fn assemble_rhs_3d<F>(mesh: &TetrahedronMesh, f: F) -> Vec<f64>
where
    F: Fn([f64; 3]) -> f64 + Sync + Send,
{
    let elem = P1Tetrahedron;
    let quad = tetrahedron_quadrature(2);
    let n_dofs = mesh.num_vertices();

    let cell_contributions: Vec<(usize, f64)> = mesh
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

            let det_j = map.det_jacobian.abs();
            let mut local_b = [0.0; 4];

            for q in &quad.points {
                let x_phys = map.map_to_physical(&verts[0], &q.point);
                let f_val = f(x_phys);
                let basis_vals = elem.evaluate_basis(&q.point);

                for i in 0..4 {
                    local_b[i] += q.weight * det_j * f_val * basis_vals[i];
                }
            }

            vec![
                (cell[0], local_b[0]),
                (cell[1], local_b[1]),
                (cell[2], local_b[2]),
                (cell[3], local_b[3]),
            ]
        })
        .collect();

    let mut b = vec![0.0; n_dofs];
    for (dof, val) in cell_contributions {
        b[dof] += val;
    }
    b
}

/// Apply Dirichlet boundary conditions using row/column elimination.
/// Modifies the system in-place so that $A_{kk} = 1$, $A_{kj} = 0$, $A_{ik} = 0$, and $b_k = g_k$.
pub fn apply_dirichlet_bc(
    mat: &SparseColMat<usize, f64>,
    rhs: &[f64],
    bc: &DirichletBC,
) -> (SparseColMat<usize, f64>, Vec<f64>) {
    let n = mat.nrows();
    let mut is_dirichlet = vec![false; n];
    let mut dirichlet_val = vec![0.0; n];

    for (&dof, &val) in bc.dofs.iter().zip(bc.values.iter()) {
        is_dirichlet[dof] = true;
        dirichlet_val[dof] = val;
    }

    let mut new_b = rhs.to_vec();

    // Adjust RHS for non-zero Dirichlet values: b_i -= A_{ik} * g_k
    // And filter matrix triplets
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

    // Set diagonal 1.0 and RHS for Dirichlet DOFs
    for (&dof, &val) in bc.dofs.iter().zip(bc.values.iter()) {
        new_triplets.push((dof, dof, 1.0));
        new_b[dof] = val;
    }

    let new_mat = SparseColMat::try_new_from_triplets(n, n, &new_triplets).unwrap();
    (new_mat, new_b)
}

/// Helper to extract all boundary DOFs for a 2D mesh.
pub fn boundary_dofs_2d(mesh: &TriangleMesh) -> Vec<usize> {
    let mut dofs: Vec<usize> = mesh
        .extract_boundary_facets()
        .iter()
        .flat_map(|f: &BoundaryFacet<2>| f.vertices)
        .collect();
    dofs.sort_unstable();
    dofs.dedup();
    dofs
}

/// Helper to extract all boundary DOFs for a 3D mesh.
pub fn boundary_dofs_3d(mesh: &TetrahedronMesh) -> Vec<usize> {
    let mut dofs: Vec<usize> = mesh
        .extract_boundary_facets()
        .iter()
        .flat_map(|f: &BoundaryFacet<3>| f.vertices)
        .collect();
    dofs.sort_unstable();
    dofs.dedup();
    dofs
}
